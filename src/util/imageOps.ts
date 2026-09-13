import type { Pt } from '../render/Overlay'

/**
 * ArcFace 家族的 112×112 五点模板，与 OpenCV SFace 的实现逐字一致
 * （opencv/modules/objdetect/src/face_recognize.cpp 里的 dst[5][2]）。
 * 顺序：影像左眼、影像右眼、鼻尖、影像左嘴角、影像右嘴角。
 */
export const ARCFACE_TEMPLATE_112: readonly (readonly [number, number])[] = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
]

/** SFace 的输入边长 */
export const FACE_SIZE = 112

/**
 * 2D 相似变换：`dst = [[a,-b],[b,a]] · src + [tx,ty]`。
 * 只有 4 个自由度（尺度、旋转、平移），没有人脸识别需要的剪切/透视。
 */
export interface Similarity {
  a: number
  b: number
  tx: number
  ty: number
}

/**
 * 最小二乘相似变换（Umeyama 在二维、限定为纯旋转时的闭式解）。
 *
 * 与 OpenCV 那套 SVD 写法结果一致：两者都取「不含反射」的最优旋转，
 * 而二维下它就是 `R = [[c,s],[-s,c]]/√(c²+s²)`，不必真的做 SVD。
 * 点集退化（重合、共线到没有面积）时返回 null。
 *
 * ⚠️ 这里的 **b 必须带负号**，别看着 `[[a,-b],[b,a]]` 对称就顺手去掉：
 * 目标函数是 `Σ|M·p_i + t − q_i|²`，p 取 src、q 取 dst，展开后 b 的系数是
 * `−2Σ(q_x·p_y − q_y·p_x)`，所以 `∂/∂b = 0 ⇒ b = −Σ(q_x·p_y − q_y·p_x)/norm`。
 * 少这个负号等价于用旋转的转置去对齐：θ 的偏头会被「再转一个 θ」放大成 2θ。
 * 实测（合成点集，把模板转 θ 后当输入，正确解必须是 R(−θ)）：
 *   θ=15° 残差 13.9px、θ=30° 26.9px、θ=45° 38.0px；而 **θ=0° 残差恒为 0**——
 * 正脸怎么试都是对的，所以这个 bug 只能靠歪头的输入暴露出来。
 */
export function estimateSimilarity(src: readonly Pt[], dst: readonly (readonly [number, number])[]): Similarity | null {
  const n = Math.min(src.length, dst.length)
  if (n < 2) return null

  let sx = 0
  let sy = 0
  let dx = 0
  let dy = 0
  for (let i = 0; i < n; i++) {
    sx += src[i].x
    sy += src[i].y
    dx += dst[i][0]
    dy += dst[i][1]
  }
  sx /= n
  sy /= n
  dx /= n
  dy /= n

  let c = 0 // Σ(dx·sx + dy·sy)
  let s = 0 // Σ(dx·sy - dy·sx)
  let norm = 0 // Σ(sx² + sy²)
  for (let i = 0; i < n; i++) {
    const px = src[i].x - sx
    const py = src[i].y - sy
    const qx = dst[i][0] - dx
    const qy = dst[i][1] - dy
    c += qx * px + qy * py
    s += qx * py - qy * px
    norm += px * px + py * py
  }
  if (norm < 1e-6) return null

  const a = c / norm
  // 负号不能省，理由见函数头注释
  const b = -s / norm

  // 尺度太小/太大说明关键点塌缩或被误检，这种对齐结果必然无意义
  const scale = Math.hypot(a, b)
  if (!Number.isFinite(scale) || scale < 1e-3 || scale > 100) return null

  return {
    a,
    b,
    tx: dx - (a * sx - b * sy),
    ty: dy - (b * sx + a * sy),
  }
}

let scratch: HTMLCanvasElement | null = null
let scratchCtx: CanvasRenderingContext2D | null = null

function ensureScratch(): CanvasRenderingContext2D {
  if (!scratchCtx) {
    scratch = document.createElement('canvas')
    scratch.width = FACE_SIZE
    scratch.height = FACE_SIZE
    // willReadFrequently：每帧都要 getImageData，不开这个在部分浏览器上会走 GPU 回读，慢很多
    const ctx = scratch.getContext('2d', { willReadFrequently: true })
    if (!ctx) throw new Error('无法创建人脸对齐用的离屏画布')
    scratchCtx = ctx
  }
  return scratchCtx
}

/**
 * 用 5 个归一化关键点把脸摆正，输出 SFace 需要的 1×3×112×112 输入张量。
 *
 * 预处理：RGB（不是 BGR）、**不做归一化**（0..255 原值）、NCHW。
 *
 * 这是实测比出来的，别凭直觉改：同一组图（Obama 官方肖像、lena、Biden、
 * 双人照）用 8 种预处理各算一遍余弦矩阵——
 *   0..255 RGB   ：同人 0.99 / 异人 0.02~0.15   ← 唯一能分开的
 *   0..1   RGB   ：异人 0.91（全挤在一起，特征退化成常数方向）
 *   (x-127.5)/128：(x-0.485)/0.229 等标准化：同样退化
 * 喂 0..1 时「同图自比」仍然是 1.0，光看自匹配完全发现不了问题，所以这里
 * 留一组异人相似度当回归基线：**异人之间必须远小于 OpenCV 的 0.363 阈值**。
 *
 * @param source    视频帧或图片元素
 * @param landmarks 归一化（0..1）的五点关键点
 * @param frameW    源画面宽（像素）
 * @param frameH    源画面高（像素）
 */
export function alignFaceToTensor(
  source: CanvasImageSource,
  landmarks: readonly Pt[],
  frameW: number,
  frameH: number,
): Float32Array | null {
  const srcPx: Pt[] = []
  for (let i = 0; i < ARCFACE_TEMPLATE_112.length; i++) {
    const p = landmarks[i]
    if (!p) return null
    srcPx.push({ x: p.x * frameW, y: p.y * frameH })
  }

  const t = estimateSimilarity(srcPx, ARCFACE_TEMPLATE_112)
  if (!t) return null

  const ctx = ensureScratch()
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  // 清成透明：对齐后落在 112×112 之外的部分取到 0，正与 OpenCV 的黑色填充等价
  ctx.clearRect(0, 0, FACE_SIZE, FACE_SIZE)
  // canvas 的变换矩阵是 [m11 m21 dx; m12 m22 dy]，对应 x' = a·x - b·y + tx
  ctx.setTransform(t.a, t.b, -t.b, t.a, t.tx, t.ty)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  try {
    ctx.drawImage(source, 0, 0)
  } catch {
    // 视频还没出帧时 drawImage 会抛 InvalidStateError
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    return null
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0)

  const { data } = ctx.getImageData(0, 0, FACE_SIZE, FACE_SIZE)
  const plane = FACE_SIZE * FACE_SIZE
  const tensor = new Float32Array(3 * plane)
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    tensor[i] = data[p] // R
    tensor[plane + i] = data[p + 1] // G
    tensor[2 * plane + i] = data[p + 2] // B
  }
  return tensor
}
