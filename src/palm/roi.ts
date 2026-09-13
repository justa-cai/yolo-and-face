import type { Pt } from '../render/Overlay'
import { warpToSquare, type Similarity } from '../util/imageOps'

/**
 * 掌纹 ROI 归一化。
 *
 * 目标和人脸那套完全同构：把掌心从画面里抠出来、摆正、缩放到固定尺寸，
 * 再交给骨干网络抽特征。这类「先归一化再比特征」的做法对姿态远比端到端模型敏感，
 * 但好处是不需要任何掌纹训练数据——用通用骨干也能凑合跑。
 *
 * ## 用哪三个解剖点
 *
 * MediaPipe 的 21 点里没有直接给「指蹼（指间谷）」，但可以用相邻两根手指的
 * 掌指关节（MCP）的中点近似：
 *
 * ```
 *        8   12  16  20        指尖
 *        |   |   |   |
 *        7   11  15  19
 *        |   |   |   |
 *        6   10  14  18
 *        |   |   |   |
 *   A →  5   9   13  17  ← B     掌指关节连线，A/B 就是两个「谷」
 *        ╲   ╲   ╱   ╱
 *         ╲   ╲ ╱   ╱
 *          ╲   ╳   ╱
 *           ╲ ╱ ╲ ╱
 *            W                    A = mid(5, 9)  B = mid(13, 17)
 *          手腕                    W = 0
 * ```
 *
 * ## 为什么不拟合三个点，而是只用两个
 *
 * 一开始是拿 A、B、W 三点做最小二乘相似变换的。实测下来这是错的：
 *
 * 二维相似变换只有 4 个自由度，而 A、B 两点就已经把「旋转 + 尺度 + 平移」全部定死
 * （第三个点唯一的用处是消掉镜像的歧义，而镜像我们另有更可靠的判据）。
 * 所以 W 点不但没提供新信息，反而把它的**测量噪声**硬塞进拟合里。
 * 更糟的是它带来的偏差是有方向的：W 到 A-B 中点的距离在 6 张实测手掌上是
 * 1.61–2.26 个 |A-B|（均值 1.98），而早期模板里写的是 0.74/0.6 = 1.23。
 * 相似变换改不了比例，最小二乘只能折中——于是正方形被放大到把
 * **手腕、前臂和背景**都框了进去，而这些恰好是「整体外观」信息最强、
 * 掌纹信息最弱的地方。实测确实因此翻车：不同人的手之间能算出 0.86 的余弦，
 * 比同一只手的旋转变体还高。
 *
 * 现在改成：**A、B 定框架，W 只用来判断手心朝哪边**。变换是精确构造的，没有残差。
 *
 * 尺度基准取 |A-B|（食指-中指谷到无名指-小指谷）。这两个点都是掌骨末端的关节位置，
 * 张合手指不会改变它们之间的距离，所以它是个稳定的「骨尺」。
 */

/** MediaPipe 手部关键点里我们用到的下标 */
const WRIST = 0
/** 食指掌指关节（食指根） */
const INDEX_MCP = 5
/** 中指掌指关节 */
const MIDDLE_MCP = 9
/** 无名指掌指关节 */
const RING_MCP = 13
/** 小指掌指关节 */
const PINKY_MCP = 17

/** 骨干网络的输入边长（两种骨干都是 224） */
export const PALM_SIZE = 224

/**
 * 正方形取多大、摆在哪，都以 |A-B| 为单位。
 *
 * 实测（6 张真实手掌，见 tmp/palm_calib.py）：
 * - 掌宽（食指掌指关节到小指掌指关节）≈ 1.5 |A-B|，且在 A-B 中点上左右对称；
 * - 掌长（掌指关节连线到手腕）≈ 1.98 |A-B|，正好是 A-B 中点到手腕沿掌轴的距离。
 *
 * 边长取 2.4，比掌长略大一点，四边留出余量吸收检测抖动；中心放在掌轴上
 * 距 A-B 中点 1.05 的地方，于是正方形在掌轴方向覆盖 -0.15 .. 2.25 个 |A-B|。
 */
const ROI_SIDE = 2.4
/** 正方形中心到 A-B 中点的距离（沿掌轴，单位 |A-B|） */
const ROI_CENTER_OFFSET = 1.05

const mid = (a: Pt, b: Pt): Pt => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 })

/**
 * 取掌心定位用的点，并统一手性。
 *
 * **手性必须处理掉**：相似变换只有旋转，不含镜像，而左手掌心和右手掌心在画面里
 * 恰好是镜像关系。若不做处理，其中一只手永远定不出正确的朝向（会退化成拧 180° 的坏解）。
 * 判据直接从关键点自己算——三角形 (A、B、W) 的有向面积——比信任 MediaPipe 的
 * `handedness` 可靠，后者在自拍镜像下会反。
 *
 * 检出「反向」的手就把 x 翻过来（x → 1-x），等价于在规范图里统一按一种手性处理。
 * 代价是左右手的特征各自成一组、互不匹配——这正是多数掌纹系统的行为，可以接受。
 */
function locate(
  landmarks: readonly Pt[],
  frameW: number,
  frameH: number,
): { a: Pt; b: Pt; wrist: Pt; flipped: boolean } | null {
  const w = landmarks[WRIST]
  const a = landmarks[INDEX_MCP]
  const b = landmarks[MIDDLE_MCP]
  const c = landmarks[RING_MCP]
  const d = landmarks[PINKY_MCP]
  if (!w || !a || !b || !c || !d) return null

  const valleyA = mid(a, b)
  const valleyB = mid(c, d)

  // 叉积符号 = 三角形 A→B→W 的绕向。手掌在画面里翻面时符号会变。
  const cross =
    (valleyB.x - valleyA.x) * (w.y - valleyA.y) - (valleyB.y - valleyA.y) * (w.x - valleyA.x)

  const flipped = cross > 0
  const toPx = (p: Pt): Pt =>
    flipped
      ? { x: (1 - p.x) * frameW, y: p.y * frameH }
      : { x: p.x * frameW, y: p.y * frameH }

  return { a: toPx(valleyA), b: toPx(valleyB), wrist: toPx(w), flipped }
}

/**
 * 掌心对齐到规范正方形的相似变换；关键点不可用或退化时返回 null。
 *
 * 变换是**精确构造**的，不是拟合出来的：A、B 两点定出旋转与尺度，
 * 手腕只负责指出手掌朝哪边（法向取哪一侧）。
 */
export function palmRoiTransform(
  landmarks: readonly Pt[],
  frameW: number,
  frameH: number,
): Similarity | null {
  const loc = locate(landmarks, frameW, frameH)
  if (!loc) return null
  const { a: A, b: B, wrist } = loc

  const dx = B.x - A.x
  const dy = B.y - A.y
  const d = Math.hypot(dx, dy)
  if (d < 1e-6) return null

  // 掌轴基向量：u 沿 A→B，n 垂直于 u。法向有两个候选，取指向手腕的那个。
  //
  // ⚠️ 必须**同时**翻 u 和 n。`Similarity` 里的 2×2 矩阵是 [[a,−b],[b,a]]，行列式恒为正，
  // 只能表达旋转、表达不了镜像；而 (u, n) 只要有一个取反就变成反射基，代进公式会算出
  // 一个 y 方向整体翻掉的坏变换（实测：正方形被挪到画面外，抠出来全黑）。
  // 两个一起取反等于把规范正方形转 180°，仍是旋转。
  const ux = dx / d
  const uy = dy / d
  let nx = -uy
  let ny = ux
  let vx = ux
  let vy = uy
  const mx = (A.x + B.x) / 2
  const my = (A.y + B.y) / 2
  if ((wrist.x - mx) * nx + (wrist.y - my) * ny < 0) {
    nx = -nx
    ny = -ny
    vx = -vx
    vy = -vy
  }

  // 正方形中心（源像素）与外接边长
  const side = ROI_SIDE * d
  const cx = mx + ROI_CENTER_OFFSET * d * nx
  const cy = my + ROI_CENTER_OFFSET * d * ny

  // 源像素 → 目标像素。`Similarity` 的约定是「源像素 → 目标像素」
  // （跟人脸那条路一致：ARCFACE_TEMPLATE_112 也是像素坐标），所以这里要乘上边长，
  // 不能吐归一化坐标——`warpToSquare` 会把它当成目标画布的像素直接用。
  //   dst.x = ((p − c)·v)/side·PALM_SIZE + PALM_SIZE/2      dst.y 同理，用 n
  // 展开成 dst = [[a,−b],[b,a]]·src + t 即得下面四个系数
  const k = PALM_SIZE / side
  const a = vx * k
  const b = nx * k
  return {
    a,
    b,
    tx: PALM_SIZE * (0.5 - (cx * vx + cy * vy) / side),
    ty: PALM_SIZE * (0.5 - (cx * nx + cy * ny) / side),
  }
}

/** 相似变换的逆：`dst → src`。规范坐标是 0..1，逆变换的结果是源画面像素。 */
function invert(t: Similarity): Similarity {
  const det = t.a * t.a + t.b * t.b
  if (det < 1e-12) return { a: 0, b: 0, tx: 0, ty: 0 }
  const a = t.a / det
  const b = -t.b / det
  // dst = M·src + t  ⇒  src = M⁻¹·(dst − t)
  return {
    a,
    b,
    tx: -(a * t.tx - b * t.ty),
    ty: -(b * t.tx + a * t.ty),
  }
}

/**
 * ROI 在**源画面**里的四个角（归一化坐标），用于在叠加层上把抠出来的区域画出来。
 *
 * 变换是按翻过手性的像素算的，所以逆映射出来的坐标也要把 x 翻回去，
 * 否则画出来的框会和实际位置左右颠倒。
 */
export function palmRoiQuad(
  landmarks: readonly Pt[],
  frameW: number,
  frameH: number,
): Pt[] | null {
  const loc = locate(landmarks, frameW, frameH)
  if (!loc) return null
  const t = palmRoiTransform(landmarks, frameW, frameH)
  if (!t) return null
  const inv = invert(t)

  // 规范正方形的四个角，单位是目标画布像素
  const corners: [number, number][] = [
    [0, 0],
    [PALM_SIZE, 0],
    [PALM_SIZE, PALM_SIZE],
    [0, PALM_SIZE],
  ]
  return corners.map(([cx, cy]) => {
    // 目标像素 → 源画面像素
    const sx = inv.a * cx - inv.b * cy + inv.tx
    const sy = inv.b * cx + inv.a * cy + inv.ty
    const nx = sx / frameW
    return { x: loc.flipped ? 1 - nx : nx, y: sy / frameH }
  })
}

/** ImageNet 的归一化参数，与两种骨干的 preprocessor_config 一致 */
const IMAGENET_MEAN = [0.485, 0.456, 0.406]
const IMAGENET_STD = [0.229, 0.224, 0.225]

/**
 * 把掌心抠成骨干要的 1×3×224×224 张量。
 *
 * 预处理：RGB → /255 → 减 ImageNet 均值除标准差 → NCHW。
 * 这套参数直接从两个模型的 `preprocessor_config.json` 抄来，别凭感觉改。
 *
 * ⚠️ 这里的归一化**不适用于人脸那条路**：SFace 是 0..255 原值进去的，
 * 加标准化反而会让特征退化（见 `alignFaceToTensor` 的注释）。两边各按各的来。
 */
export function palmRoiToTensor(
  source: CanvasImageSource,
  landmarks: readonly Pt[],
  frameW: number,
  frameH: number,
): Float32Array | null {
  const t = palmRoiTransform(landmarks, frameW, frameH)
  if (!t) return null

  const img = warpToSquare(source, t, PALM_SIZE)
  if (!img) return null

  const { data } = img
  const plane = PALM_SIZE * PALM_SIZE
  const tensor = new Float32Array(3 * plane)
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    for (let c = 0; c < 3; c++) {
      tensor[c * plane + i] = (data[p + c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    }
  }
  return tensor
}
