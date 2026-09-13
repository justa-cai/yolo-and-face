import type { AssetSpec } from '../tasks/types'
import { loadAsset, wasmDir } from './assets'

/**
 * onnxruntime-web 的类型。模块本身走动态 import（见 loadOrt），
 * 这里只取类型，不会把运行时打进主 chunk。
 */
export type OrtModule = typeof import('onnxruntime-web')
type OrtSession = import('onnxruntime-web').InferenceSession

/** ORT 的执行后端。WebGPU 只在真有可用适配器时才会被选中。 */
export type OrtBackend = 'webgpu' | 'wasm'

let ortPromise: Promise<OrtModule> | null = null
/** 当前 ortPromise 用的是带 WebGPU 的入口还是纯 CPU 入口 */
let ortIsWebgpuBundle = false
let configured = false
let backendLabel: string | null = null
/** WebGPU 运行时可用性的探测结果缓存，null 表示还没探过 */
let webgpuRuntime: boolean | null = null

/**
 * 加载 onnxruntime-web 并把运行时目录指到自托管的 public/wasm/ort/。
 *
 * 两个要点：
 *
 * 1. 用动态 import 而不是顶层 import：ORT 的 JS 胶水层本身就有几百 KB，
 *    顶层 import 会让它在页面首屏就被打进主 chunk，违背「按算法懒加载」。
 * 2. ORT 按执行后端拆了 bundle，而**默认入口**（`onnxruntime-web`）固定去取
 *    `ort-wasm-simd-threaded.jsep.mjs` 这份 WebGPU 版胶水层，即使你只要
 *    WASM 后端也一样 —— 默认的 `pnpm fetch-assets` 并不拷它，于是建会话直接
 *    404。所以只有确认 jsep 运行时也部署了（`--with-webgpu`）才用默认入口，
 *    否则用纯 CPU 的 `onnxruntime-web/wasm`。两者 JS 本身都不大，动态加载
 *    时只有被选中的那个会真的下载。
 */
function loadOrtBundle(webgpu: boolean): Promise<OrtModule> {
  if (ortPromise && ortIsWebgpuBundle === webgpu) return ortPromise
  ortIsWebgpuBundle = webgpu
  ortPromise = (webgpu ? import('onnxruntime-web') : import('onnxruntime-web/wasm'))
    .then((mod) => {
      // 打包器不同形态下命名空间可能挂在 default 上，统一一下
      const ort = ((mod as unknown as { default?: OrtModule }).default ?? mod) as OrtModule
      configureEnv(ort)
      return ort
    })
    .catch((err) => {
      // 失败就别把坏 promise 缓存下来，下次还能重试
      ortPromise = null
      throw err
    })
  return ortPromise
}

/** 已经加载好的那个模块；没加载过时按纯 CPU 入口加载（与默认自托管资产一致） */
export function loadOrt(): Promise<OrtModule> {
  return loadOrtBundle(ortPromise ? ortIsWebgpuBundle : false)
}

function configureEnv(ort: OrtModule): void {
  if (configured) return
  ort.env.wasm.wasmPaths = wasmDir('ort')
  // SIMD + 多线程版的 WASM 需要 SharedArrayBuffer，而它只在跨源隔离
  // （响应头带 COOP/COEP）时才可用。内网的静态服务器通常不会配这两个头，
  // 这时强行多线程会在初始化阶段直接抛错，所以默认单线程。
  const cores = navigator.hardwareConcurrency || 1
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, cores) : 1
  ort.env.logLevel = 'error'
  configured = true
}

interface GpuLike {
  requestAdapter(): Promise<unknown | null>
}

/**
 * WebGPU 版（jsep）运行时是否随站点一起部署了。
 *
 * 默认的 `pnpm fetch-assets` 只拷纯 WASM 那两份，jsep 要 `--with-webgpu` 才带上
 * （+28MB）。如果没部署却仍去建 WebGPU 会话，ORT 会在请求 jsep 胶水层时 404，
 * 虽然能靠下面的 catch 退回 wasm，但每次首屏都刷一条吓人的报错，所以先探一下。
 */
async function webgpuRuntimeAvailable(): Promise<boolean> {
  if (webgpuRuntime !== null) return webgpuRuntime
  try {
    const res = await fetch(`${wasmDir('ort')}ort-wasm-simd-threaded.jsep.mjs`, { method: 'HEAD' })
    // 不能只看 res.ok：静态服务器对不存在的路径可能回 200 + index.html
    // （Vite dev 与带 SPA 兜底的 nginx 都这样），得确认回的真是 JS
    const type = res.headers.get('content-type') ?? ''
    webgpuRuntime = res.ok && !type.includes('text/html')
  } catch {
    webgpuRuntime = false
  }
  return webgpuRuntime
}

/**
 * 选执行后端。
 *
 * 先看 jsep 运行时在不在 —— 没部署就只有 WASM 一条路，连 `navigator.gpu`
 * 都不必碰（软件渲染/无头环境里 Chromium 一访问它就刷一堆 Dawn 报错）。
 * 再探适配器：有 `navigator.gpu` 不等于能用，老显卡、无头浏览器、软件渲染下
 * `requestAdapter()` 会返回 null，这时候必须退回 wasm，否则建会话直接失败。
 */
export async function pickOrtBackend(): Promise<OrtBackend> {
  if (!(await webgpuRuntimeAvailable())) return 'wasm'
  const gpu = (navigator as Navigator & { gpu?: GpuLike }).gpu
  if (!gpu) return 'wasm'
  try {
    const adapter = await gpu.requestAdapter()
    if (!adapter) return 'wasm'
    return 'webgpu'
  } catch {
    return 'wasm'
  }
}

export interface OrtSessionHandle {
  session: OrtSession
  backend: OrtBackend
}

/**
 * 加载模型字节并创建推理会话，优先 WebGPU，失败自动退回 WASM。
 *
 * 与 MediaPipe 那条路径同构：模型字节由我们自己拉（可报进度、可离线缓存），
 * 再以 buffer 交给 ORT，完全不依赖它自己的网络加载路径。
 */
export async function createOrtSession(
  spec: AssetSpec,
  report: (text: string, fraction: number) => void,
): Promise<OrtSessionHandle> {
  report('加载推理运行时', 0.05)
  const preferred = await pickOrtBackend()
  const ort = await loadOrtBundle(preferred === 'webgpu')

  report('下载模型', 0.1)
  const buffer = await loadAsset(spec, (f) => {
    if (f !== null) report('下载模型', 0.1 + f * 0.85)
  })

  report('初始化', 0.96)
  try {
    const session = await ort.InferenceSession.create(buffer, {
      executionProviders: [preferred],
      graphOptimizationLevel: 'all',
    })
    backendLabel = preferred === 'webgpu' ? 'ONNX WebGPU' : 'ONNX WASM'
    return { session, backend: preferred }
  } catch (err) {
    if (preferred === 'wasm') throw err
    console.warn(`[ort] WebGPU 后端初始化失败，回退 WASM：`, err)
    // 回退要换回纯 CPU 入口：WebGPU 入口只认 jsep 那份胶水层，
    // 它既然不可用，同一个模块里也没有可用的 wasm 后端
    const cpuOrt = await loadOrtBundle(false)
    const session = await cpuOrt.InferenceSession.create(buffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
    backendLabel = 'ONNX WASM'
    return { session, backend: 'wasm' }
  }
}

/** 已建好的 ORT 后端名，用于性能面板展示；还没建会话时为 null */
export function currentOrtLabel(): string | null {
  return backendLabel
}
