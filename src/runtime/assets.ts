import type { AssetSpec } from '../tasks/types'

const CACHE_NAME = 'cv-models-v1'

/**
 * 把站点内的相对路径解析成绝对 URL。
 *
 * 必须以 document.baseURI 为基准，而不是以 '/' 为基准：同一份产物既要能在
 * 内网静态服务器上以根目录打开，又要能挂在 GitHub Pages 的 /<repo>/ 子路径下。
 * `${import.meta.env.BASE_URL}` 在 base:'./' 时是 './'，拼出来仍是相对路径，
 * MediaPipe 的 FilesetResolver 不接受相对路径，所以这里统一转成绝对 URL。
 */
export function siteUrl(relativePath: string): string {
  return new URL(relativePath, document.baseURI).href
}

/** MediaPipe / ORT 的 WASM 运行时目录（绝对 URL，结尾带 /） */
export function wasmDir(name: 'mediapipe' | 'ort'): string {
  return siteUrl(`wasm/${name}/`)
}

/** 单个资产的下载进度回调：fraction 为 0..1，总长未知时为 null */
export type ProgressFn = (fraction: number | null) => void

export function formatBytes(n: number): string {
  if (n >= 1048576) return `${(n / 1048576).toFixed(1)} MB`
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${n} B`
}

function cacheAvailable(): boolean {
  return typeof caches !== 'undefined' && 'open' in caches
}

/**
 * 按需加载一个模型资产。
 *
 * 刻意不走 MediaPipe / ORT 自己的加载路径，而是自己取字节再喂 buffer：
 * 1. 能报下载进度（对 36MB 的 SFace 尤其必要）；
 * 2. 能显式写进 Cache Storage，二次访问完全不走网络，断网也能用；
 * 3. 不依赖 HTTP 缓存的响应头，静态服务器怎么配都行。
 */
export async function loadAsset(spec: AssetSpec, onProgress?: ProgressFn): Promise<ArrayBuffer> {
  const url = siteUrl(spec.url)
  const cache = cacheAvailable() ? await caches.open(CACHE_NAME) : null

  if (cache) {
    const hit = await cache.match(url)
    if (hit) {
      onProgress?.(1)
      const buf = await hit.arrayBuffer()
      if (buf.byteLength > 0) return buf
    }
  }

  const res = await fetch(url)
  if (!res.ok) throw new Error(`${spec.label} 下载失败：HTTP ${res.status}`)

  const total = Number(res.headers.get('content-length')) || spec.bytes || 0
  const buffer = await readWithProgress(res, total, onProgress)

  // 写缓存失败不是致命错误（例如隐私模式或配额不足），照常把 buffer 交出去
  if (cache) {
    try {
      await cache.put(url, new Response(buffer.slice(0)))
    } catch {
      /* 忽略 */
    }
  }

  return buffer
}

async function readWithProgress(
  res: Response,
  total: number,
  onProgress?: ProgressFn,
): Promise<ArrayBuffer> {
  if (!res.body) {
    const buf = await res.arrayBuffer()
    onProgress?.(1)
    return buf
  }

  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (value) {
      chunks.push(value)
      received += value.byteLength
      onProgress?.(total > 0 ? Math.min(1, received / total) : null)
    }
  }

  const out = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.byteLength
  }
  onProgress?.(1)
  return out.buffer
}

/** 并发加载一组资产，进度按总字节数加权汇总。 */
export async function loadAssets(
  specs: readonly AssetSpec[],
  onProgress?: ProgressFn,
): Promise<Map<string, ArrayBuffer>> {
  const out = new Map<string, ArrayBuffer>()
  if (specs.length === 0) {
    onProgress?.(1)
    return out
  }

  const loaded = new Map<string, number>()
  const totals = new Map<string, number>()
  const report = (): void => {
    let have = 0
    let all = 0
    for (const s of specs) {
      all += totals.get(s.url) ?? s.bytes ?? 0
      have += loaded.get(s.url) ?? 0
    }
    onProgress?.(all > 0 ? Math.min(1, have / all) : null)
  }

  await Promise.all(
    specs.map(async (s) => {
      const buf = await loadAsset(s, (f) => {
        if (f === null) return
        const size = totals.get(s.url) ?? s.bytes ?? 0
        loaded.set(s.url, size * f)
        report()
      })
      totals.set(s.url, buf.byteLength)
      loaded.set(s.url, buf.byteLength)
      report()
      out.set(s.url, buf)
    }),
  )

  onProgress?.(1)
  return out
}

/** 清空模型缓存（UI 上的「清理缓存」用） */
export async function clearAssetCache(): Promise<void> {
  if (!cacheAvailable()) return
  await caches.delete(CACHE_NAME)
}
