#!/usr/bin/env node
/**
 * 把所有模型与 WASM 运行时拉到 public/ 下，之后整个站点可完全离线运行。
 *
 *   pnpm fetch-assets                 # 默认：最小可用集（MediaPipe SIMD + ORT 纯 WASM）
 *   pnpm fetch-assets --all-wasm      # 额外拉 MediaPipe 的 nosimd / 多线程版
 *   pnpm fetch-assets --with-webgpu   # 额外拉 ORT 的 jsep 版（WebGPU/WebGL 后端需要，+28MB）
 *   pnpm fetch-assets --force         # 忽略本地已存在，全部重下
 *
 * 幂等：目标文件已存在且字节数与远端一致时跳过。
 * 例外是掌纹骨干 MobileNetV3——它下载后还要打补丁（见 PATCHED_MOBILENET），
 * 判定依据换成「打完补丁的字节数」，原始图本身不留档。
 */
import { mkdir, stat, copyFile, readdir, unlink } from 'node:fs/promises'
import { createWriteStream } from 'node:fs'
import { pipeline } from 'node:stream/promises'
import { Readable } from 'node:stream'
import { spawn } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const PUBLIC_DIR = path.join(ROOT, 'public')
/** 中间产物一律落在这里，下载成功后由调用方删掉 */
const SCRATCH_DIR = path.join(ROOT, 'tmp', 'download')

const argv = new Set(process.argv.slice(2))
const ALL_WASM = argv.has('--all-wasm')
const WITH_WEBGPU = argv.has('--with-webgpu')
const FORCE = argv.has('--force')

const MP = 'https://storage.googleapis.com/mediapipe-models'
const ZOO = 'https://github.com/opencv/opencv_zoo/raw/main/models'
const HF = 'https://huggingface.co'

/** 远端模型 -> public/ 下的相对路径 */
const MODELS = [
  // --- MediaPipe Tasks Vision（Apache-2.0）---
  ['face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
   'models/mediapipe/blaze_face_short_range.tflite'],
  ['face_landmarker/face_landmarker/float16/1/face_landmarker.task',
   'models/mediapipe/face_landmarker.task'],
  ['pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
   'models/mediapipe/pose_landmarker_lite.task'],
  ['pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
   'models/mediapipe/pose_landmarker_full.task'],
  ['object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
   'models/mediapipe/efficientdet_lite0.tflite'],
  ['image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
   'models/mediapipe/efficientnet_lite0.tflite'],
  // 手部 21 点，掌纹识别的前置（Apache-2.0）
  ['hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
   'models/mediapipe/hand_landmarker.task'],
].map(([rel, dest]) => ({ url: `${MP}/${rel}`, dest }))

/** OpenCV Zoo 走 GitHub raw，URL 形态不同，单独追加 */
MODELS.push({
  url: `${ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx`,
  dest: 'models/onnx/face_recognition_sface_2021dec.onnx',
})

/**
 * 掌纹骨干。两个都是 Apache-2.0 的通用视觉骨干（**不是**掌纹专用模型，
 * 见 src/palm/backbone.ts 的说明），走 HuggingFace 的 onnx-community 镜像。
 */
MODELS.push({
  url: `${HF}/onnx-community/dinov2-small-ONNX/resolve/main/onnx/model_fp16.onnx`,
  dest: 'models/onnx/palm_dinov2_small_fp16.onnx',
})

/**
 * MobileNetV3 不能直接用：官方导出只有 `logits`（1000 维 ImageNet 分类分数）这一个输出，
 * 拿分类分数当特征向量是很差的选择。所以先下原始图，再用 scripts/patch-onnx-output.py
 * 把分类头摘掉、改成暴露分类前的 1024 维池化特征。
 *
 * `expectBytes` 是打完补丁后的字节数，用来做幂等判断——补丁脚本是确定性的
 * （只删节点/权重、加一条输出声明，不做任何量化），同样的输入必然得到同样的输出。
 */
const PATCHED_MOBILENET = {
  srcUrl: `${HF}/onnx-community/mobilenetv3_small_100.lamb_in1k/resolve/main/onnx/model.onnx`,
  /** 原始图下载到这里，打完补丁就删 */
  raw: path.join(SCRATCH_DIR, 'mobilenetv3_small_100.lamb_in1k.onnx'),
  dest: 'models/onnx/palm_mobilenetv3_features.onnx',
  expectBytes: 6104239,
}

/** 从 node_modules 拷贝的 WASM 运行时： [源目录, 目标目录, 文件名谓词] */
const WASM_COPIES = [
  {
    from: ['@mediapipe', 'tasks-vision', 'wasm'],
    to: 'wasm/mediapipe',
    pick: (f) => ALL_WASM || f.startsWith('vision_wasm_internal.'),
  },
  {
    from: ['onnxruntime-web', 'dist'],
    to: 'wasm/ort',
    // ORT 会按需请求 `${wasmPaths}ort-wasm-simd-threaded[.jsep].mjs|.wasm`
    pick: (f) =>
      WITH_WEBGPU
        ? /^ort-wasm-simd-threaded(\.jsep)?\.(mjs|wasm)$/.test(f)
        : /^ort-wasm-simd-threaded\.(mjs|wasm)$/.test(f),
  },
]

const mb = (n) => `${(n / 1048576).toFixed(2)} MB`
const rel = (p) => path.relative(ROOT, p)

/** 远端内容的字节数；失败返回 null */
async function remoteSize(url) {
  try {
    // GitHub raw 对 HEAD 不稳定，统一用 GET + Range 取尾部
    const res = await fetch(url, { method: 'HEAD', redirect: 'follow' })
    if (!res.ok) return null
    const len = res.headers.get('content-length')
    return len ? Number(len) : null
  } catch {
    return null
  }
}

async function localSize(p) {
  try {
    return (await stat(p)).size
  } catch {
    return null
  }
}

async function download(url, dest) {
  await mkdir(path.dirname(dest), { recursive: true })

  if (!FORCE) {
    const [ls, rs] = await Promise.all([localSize(dest), remoteSize(url)])
    if (ls !== null && (rs === null || ls === rs)) {
      console.log(`  skip   ${rel(dest)}  (${mb(ls)})`)
      return { skipped: true, size: ls }
    }
  }

  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${url}`)

  // 先写 .part 再改名，避免中断留下半个文件被下次误判为"已存在"
  const part = `${dest}.part`
  await pipeline(Readable.fromWeb(res.body), createWriteStream(part))
  const { rename } = await import('node:fs/promises')
  await rename(part, dest)

  const size = (await localSize(dest)) ?? 0
  console.log(`  get    ${rel(dest)}  (${mb(size)})`)
  return { skipped: false, size }
}

/** 跑一次 python 补丁脚本；非零退出即抛错 */
function runPatch(src, dst) {
  return new Promise((resolve, reject) => {
    const py = process.env.PYTHON ?? 'python3'
    const script = path.join(ROOT, 'scripts', 'patch-onnx-output.py')
    const child = spawn(py, [script, src, dst], { stdio: 'inherit' })
    child.on('error', (err) =>
      reject(new Error(`调用 ${py} 失败（${err.message}）。补丁需要 onnx 包：pip install onnx`)),
    )
    child.on('exit', (code) =>
      code === 0 ? resolve() : reject(new Error(`补丁脚本退出码 ${code}`)),
    )
  })
}

/**
 * 下载 MobileNetV3 原始图 -> 打补丁 -> 落盘成「带 1024 维特征输出」的版本。
 * 目标文件已经是打过补丁的（字节数对得上）就整个跳过，连原始图都不下。
 */
async function fetchPatchedMobilenet() {
  const { srcUrl, raw, dest, expectBytes } = PATCHED_MOBILENET
  const destPath = path.join(PUBLIC_DIR, dest)

  if (!FORCE) {
    const ls = await localSize(destPath)
    if (ls === expectBytes) {
      console.log(`  skip   ${rel(destPath)}  (${mb(ls)}，已打过补丁)`)
      return ls
    }
  }

  await download(srcUrl, raw)
  await mkdir(path.dirname(destPath), { recursive: true })
  await runPatch(raw, destPath)
  await unlink(raw).catch(() => {})

  const size = (await localSize(destPath)) ?? 0
  if (size !== expectBytes) {
    console.warn(
      `  ⚠ ${rel(destPath)} 打完补丁是 ${size} 字节，与预期的 ${expectBytes} 不符。` +
        `模型结构可能变了，请确认 $TAP 层，并同步更新 src/palm/backbone.ts 里的 bytes。`,
    )
  }
  return size
}

async function copyWasm() {  let total = 0
  for (const { from, to, pick } of WASM_COPIES) {
    const srcDir = path.join(ROOT, 'node_modules', ...from)
    const dstDir = path.join(PUBLIC_DIR, to)
    await mkdir(dstDir, { recursive: true })

    let entries
    try {
      entries = (await readdir(srcDir)).filter(pick)
    } catch {
      throw new Error(`找不到 ${rel(srcDir)}，请先执行 pnpm install`)
    }
    if (entries.length === 0) throw new Error(`${rel(srcDir)} 下没有匹配的运行时文件`)

    for (const name of entries) {
      const src = path.join(srcDir, name)
      const dst = path.join(dstDir, name)
      const [ss, ds] = await Promise.all([localSize(src), localSize(dst)])
      if (!FORCE && ss === ds) {
        console.log(`  skip   ${rel(dst)}  (${mb(ds)})`)
        total += ds
        continue
      }
      await copyFile(src, dst)
      console.log(`  copy   ${rel(dst)}  (${mb(ss)})`)
      total += ss ?? 0
    }

    // 清掉开关切换后残留的旧文件（例如 --with-webgpu 拉过 jsep 后又跑默认模式）
    const stale = (await readdir(dstDir)).filter((f) => !pick(f) && !f.includes('.part'))
    for (const f of stale) {
      await unlink(path.join(dstDir, f))
      console.log(`  prune  ${rel(path.join(dstDir, f))}`)
    }
  }
  return total
}

async function main() {
  if (ALL_WASM && WITH_WEBGPU) console.log('（已启用全部 WASM 变体）')
  console.log(`\n下载模型 -> public/models/`)
  let modelBytes = 0
  for (const { url, dest } of MODELS) {
    const { size } = await download(url, path.join(PUBLIC_DIR, dest))
    modelBytes += size ?? 0
  }

  // MobileNetV3 要过一道补丁，单独走
  modelBytes += await fetchPatchedMobilenet()

  console.log(`\n拷贝 WASM 运行时 -> public/wasm/`)
  const wasmBytes = await copyWasm()

  console.log(
    `\n完成。模型 ${mb(modelBytes)} + 运行时 ${mb(wasmBytes)} = 合计 ${mb(modelBytes + wasmBytes)}`
  )
  console.log(`站点资源已全部落在 public/，从此可断网运行。\n`)
}

main().catch((err) => {
  console.error(`\n失败：${err.message}\n`)
  process.exit(1)
})
