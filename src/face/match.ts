/**
 * 人脸特征向量的匹配。
 *
 * 度量方式与 OpenCV `FaceRecognizerSF` 一致：先 L2 归一化再点积，即余弦相似度；
 * 判定同一人的默认阈值 0.363 也沿用 opencv_zoo 的取值（见 face_recognition_sface/sface.py）。
 * 余弦对向量长度不敏感，所以模型输出的模长不需要额外处理。
 */

export interface FaceEntry {
  id: string
  name: string
  /** 128 维特征向量（SFace 输出） */
  embedding: Float32Array
  /** 注册时间（毫秒） */
  createdAt: number
}

export interface FaceMatch {
  id: string
  name: string
  /** 余弦相似度，越大越像 */
  score: number
}

/** SFace 的官方判定阈值：余弦 ≥ 0.363 视为同一人 */
export const DEFAULT_THRESHOLD = 0.363

export function l2Normalize(v: Float32Array): Float32Array {
  let sum = 0
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i]
  const len = Math.sqrt(sum)
  const out = new Float32Array(v.length)
  if (len < 1e-12) return out
  for (let i = 0; i < v.length; i++) out[i] = v[i] / len
  return out
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length)
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom < 1e-12 ? 0 : dot / denom
}

/**
 * 在人脸库里找最像的若干条。
 *
 * `topK` 默认 1（只要最佳匹配）；调用方可以调大来观察「第二、第三像的是谁」，
 * 调阈值时这比只看一个分数有用得多。
 */
export function matchFace(
  query: Float32Array,
  gallery: readonly FaceEntry[],
  topK = 1,
): FaceMatch[] {
  const scored: FaceMatch[] = gallery.map((e) => ({
    id: e.id,
    name: e.name,
    score: cosineSimilarity(query, e.embedding),
  }))
  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, Math.max(1, topK))
}
