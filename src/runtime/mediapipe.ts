import { FilesetResolver } from '@mediapipe/tasks-vision'
import { loadAsset, wasmDir } from './assets'
import type { AssetSpec } from '../tasks/types'

/** 推理后端。GPU 走 WebGL，CPU 走 WASM。 */
export type Delegate = 'GPU' | 'CPU'

/** tasks-vision 没有导出 WasmFileset 类型，从 forVisionTasks 的返回值反推 */
export type VisionFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>

let filesetPromise: Promise<VisionFileset> | null = null

/** WebGL 渲染器探测结果，只算一次 */
let delegateCache: Delegate | null = null

/**
 * 选一个合适的 delegate。
 *
 * 不是"有 WebGL 就用 WebGL"：软件渲染（SwiftShader / llvmpipe / 无硬件加速的
 * 无头浏览器）下 GPU 路径比 WASM 还慢一个数量级，MediaPipe 在人脸上要跑几百毫秒，
 * 那种情况必须退回 CPU。所以这里读一下真实的渲染器名字再决定。
 */
export function pickDelegate(): Delegate {
  if (delegateCache) return delegateCache

  let delegate: Delegate = 'CPU'
  try {
    const canvas = document.createElement('canvas')
    const gl = (canvas.getContext('webgl2') ?? canvas.getContext('webgl')) as WebGLRenderingContext | null
    if (gl) {
      const ext = gl.getExtension('WEBGL_debug_renderer_info')
      const name = String(
        ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      )
      const software = /swiftshader|llvmpipe|softwarerasterizer|basic render|mesa offscreen/i.test(name)
      delegate = software ? 'CPU' : 'GPU'
      if (software) console.info(`[mediapipe] 检测到软件渲染（${name}），改用 CPU(WASM) 后端`)
      // 探测用的上下文立刻释放，别占着一个 WebGL slot
      gl.getExtension('WEBGL_lose_context')?.loseContext()
    }
  } catch {
    delegate = 'CPU'
  }

  delegateCache = delegate
  return delegate
}

/**
 * MediaPipe Vision 的 WASM 运行时单例。
 *
 * 指向自托管的 public/wasm/mediapipe/，全程不碰 CDN；多个任务共用同一份
 * 运行时，只有第一次会真正去取。
 *
 * 只用 `vision_wasm_internal.*`（SIMD、单线程）这一份。`FilesetResolver` 的
 * 第二参数 `useModule` 看着像是「开多线程」，实测是坑，别再试：
 * 它选的是 `vision_wasm_module_internal.js`，那是一份 **ES module**，而
 * MediaPipe 1.0.1 在主线程上是用 `<script>` 标签去执行它的（只有 Worker 里
 * 才会走 dynamic import），于是直接 `SyntaxError: Cannot use 'import.meta'
 * outside a module`，随后报 `ModuleFactory not set`。
 * 换句话说那两个变体的差别是**打包形态**（给 Worker 用的 ESM）而不是线程数——
 * 拆包看过，两份 wasm 里的 pthread/SAB 符号一模一样。
 * 真要吃多线程，得把任务整体搬进 Web Worker，那是另一个量级的改动（见 README）。
 */
export function loadVisionFileset(): Promise<VisionFileset> {
  filesetPromise ??= FilesetResolver.forVisionTasks(wasmDir('mediapipe')).catch((err) => {
    // 失败就别把坏 promise 缓存下来，下次还能重试
    filesetPromise = null
    throw err
  })
  return filesetPromise
}

/**
 * 加载模型并创建 MediaPipe 任务，优先 GPU(WebGL)，失败自动退回 CPU(WASM)。
 *
 * 模型字节由我们自己拉（可报进度、可离线缓存），再以 buffer 形式交给 MediaPipe，
 * 因此完全不依赖 MediaPipe 自己的网络加载路径。
 */
export async function createVisionTask<T>(
  spec: AssetSpec,
  report: (text: string, fraction: number) => void,
  make: (fileset: VisionFileset, modelAssetBuffer: Uint8Array, delegate: Delegate) => Promise<T>,
): Promise<T> {
  report('加载推理运行时', 0.05)
  const fileset = await loadVisionFileset()

  report('下载模型', 0.1)
  const buffer = await loadAsset(spec, (f) => {
    if (f !== null) report('下载模型', 0.1 + f * 0.8)
  })
  const modelAssetBuffer = new Uint8Array(buffer)

  const preferred = pickDelegate()
  try {
    report('初始化', 0.95)
    return await make(fileset, modelAssetBuffer, preferred)
  } catch (err) {
    if (preferred === 'CPU') throw err
    console.warn(`[mediapipe] ${preferred} 后端初始化失败，回退 CPU：`, err)
    report('回退 CPU 后端', 0.97)
    return await make(fileset, modelAssetBuffer, 'CPU')
  }
}

/** 当前推理后端的名字，用于顶栏展示 */
export function currentBackendLabel(): string {
  return pickDelegate() === 'GPU' ? 'WASM + WebGL' : 'WASM (CPU)'
}
