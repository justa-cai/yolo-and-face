import { defineConfig } from 'vite'

/**
 * 跨源隔离所需的两个响应头。
 *
 * 有了它们，`crossOriginIsolated === true`，SharedArrayBuffer 才可用：
 * MediaPipe 能切到多线程的 `vision_wasm_module_internal`，ORT 也能把
 * `numThreads` 开到 4。实测对物体检测这种重任务的提升最大。
 *
 * 内网 nginx / 静态服务器请照抄这两个头；**GitHub Pages 配不了**，
 * 那边会自动退回单线程——功能照常，只是慢一些，所以这只是增强项不是必需项。
 */
export const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

/**
 * onnxruntime-web 的胶水层里写着 `new URL('ort-wasm-simd-threaded.wasm', import.meta.url)`，
 * 打包器会老老实实把这两份 wasm（合计约 42MB）也塞进 dist/assets/。
 * 但我们运行时把 `ort.env.wasm.wasmPaths` 指到了自托管的 public/wasm/ort/，
 * 这两份产物**永远不会被请求**——纯粹是白占体积。
 * 这里把它们解析成一个空模块，产物里就不再出现。
 */
function dropBundledOrtWasm() {
  return {
    name: 'drop-bundled-ort-wasm',
    // rollup/rolldown 的资产管线在 generateBundle 之前就把它们收进 bundle 了，
    // resolveId/load 拦不住，所以直接在产物里摘掉。
    generateBundle(_options: unknown, bundle: Record<string, unknown>) {
      for (const key of Object.keys(bundle)) {
        if (/ort-wasm-[^/]*\.wasm$/.test(key)) {
          delete bundle[key]
          this.info(`已剔除未使用的 ORT wasm 产物：${key}`)
        }
      }
    },
  }
}

export default defineConfig({
  // 相对路径：同一份产物既能挂在 GitHub Pages 的子路径下，
  // 也能直接被内网任意静态服务器（含 `python3 -m http.server`）以根目录打开。
  base: './',
  server: {
    host: true,
    port: 5173,
    headers: crossOriginIsolationHeaders,
  },
  plugins: [dropBundledOrtWasm()],
  build: {
    target: 'es2022',
    // public/ 下的模型与 wasm 是整目录拷贝，不做 transform，这里只影响 import 进来的资源
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 2048,
  },
})
