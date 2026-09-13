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
 */
import { mkdir, stat, copyFile, readdir, unlink } from 'node:fs/promises'
import { createWriteStream } from 'node:fs'
import { pipeline } from 'node:stream/promises'
import { Readable } from 'node:stream'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const PUBLIC_DIR = path.join(ROOT, 'public')

const argv = new Set(process.argv.slice(2))
const ALL_WASM = argv.has('--all-wasm')
const WITH_WEBGPU = argv.has('--with-webgpu')
const FORCE = argv.has('--force')

const MP = 'https://storage.googleapis.com/mediapipe-models'
const ZOO = 'https://github.com/opencv/opencv_zoo/raw/main/models'

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
].map(([rel, dest]) => ({ url: `${MP}/${rel}`, dest }))

/** OpenCV Zoo 走 GitHub raw，URL 形态不同，单独追加 */
MODELS.push({
  url: `${ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx`,
  dest: 'models/onnx/face_recognition_sface_2021dec.onnx',
})

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

async function copyWasm() {
  let total = 0
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
