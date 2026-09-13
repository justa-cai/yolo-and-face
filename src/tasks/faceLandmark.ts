import { FaceLandmarker, type FaceLandmarkerResult } from '@mediapipe/tasks-vision'
import type { Overlay, Pt, Rect } from '../render/Overlay'
import { createVisionTask } from '../runtime/mediapipe'
import type { AssetSpec, FrameCtx, TaskOptionSpec, TaskOptionValue, VisionTask } from './types'

/** 供人脸识别任务取用：最近一帧每张脸的 478 个归一化关键点 */
export const SHARED_FACE_LANDMARKS = 'face.landmarks'
/** 供人脸识别任务取用：最近一帧每张脸归一化后的外接框 */
export const SHARED_FACE_BOXES = 'face.boxes'

/** 鼻尖与两个嘴角的下标（478 点里） */
const NOSE_TIP = 1
const MOUTH_CORNERS = [61, 291] as const

/**
 * 取用于五点仿射对齐的五个点，顺序与 ArcFace 模板一致
 * （影像左眼、影像右眼、鼻尖、影像左嘴角、影像右嘴角）。左右一律按 x 排序，
 * 这样镜像过的画面也不会把左右眼对调。
 *
 * 眼位取的是**虹膜中心**而不是 33/263 这种外眼角：ArcFace 模板里的两个眼点
 * 是瞳孔，用外眼角会让瞳距虚大到 1.4 倍，最小二乘为了同时照顾眼睛和嘴角，
 * 会把整张脸放大并拉歪——同一个人换个角度，余弦相似度就从 0.7 掉到 0.3 上下。
 * （实测：lena 的虹膜间距 / 眼-嘴垂距 = 0.76，与模板的 0.87 接近；用外眼角
 * 是 1.07，差了 24%。）
 */
export function fivePointLandmarks(landmarks: readonly Pt[]): Pt[] | null {
  const eyeA = irisCenter(landmarks, LEFT_IRIS)
  const eyeB = irisCenter(landmarks, RIGHT_IRIS)
  const nose = landmarks[NOSE_TIP]
  const mouthA = landmarks[MOUTH_CORNERS[0]]
  const mouthB = landmarks[MOUTH_CORNERS[1]]
  if (!eyeA || !eyeB || !nose || !mouthA || !mouthB) return null

  const left = eyeA.x <= eyeB.x ? eyeA : eyeB
  const right = eyeA.x <= eyeB.x ? eyeB : eyeA
  const m0 = mouthA.x <= mouthB.x ? mouthA : mouthB
  const m1 = mouthA.x <= mouthB.x ? mouthB : mouthA
  return [left, right, nose, m0, m1]
}

function irisCenter(landmarks: readonly Pt[], iris: readonly number[]): Pt | null {
  let x = 0
  let y = 0
  let n = 0
  for (const i of iris) {
    const p = landmarks[i]
    if (!p) continue
    x += p.x
    y += p.y
    n++
  }
  return n === 0 ? null : { x: x / n, y: y / n }
}

const MODEL: AssetSpec = {
  url: 'models/mediapipe/face_landmarker.task',
  label: '人脸关键点模型',
  bytes: 3758596,
}

/** 虹膜与瞳孔在 478 点里的下标 */
const LEFT_IRIS = [468, 469, 470, 471, 472]
const RIGHT_IRIS = [473, 474, 475, 476, 477]

type PointMode = 'all' | 'contour' | 'none'
type MeshMode = 'off' | 'features' | 'tess'
type BlendMode = 'off' | 'top3'

export class FaceLandmarkTask implements VisionTask {
  readonly id = 'face-landmark'
  readonly name = '人脸关键点'
  readonly stage = 'landmark' as const
  readonly hint = '478 个 3D 关键点，含虹膜与 52 项表情系数'
  readonly assets = [MODEL]
  /**
   * 默认推理间隔。全开时几个任务会叠在同一帧里排队，帧时间直接变成它们的耗时之和，
   * 所以这里给每个任务定一个「够用就好」的重算频率：画面照 60fps 画，结果按各自的
   * 节奏更新。静态图不走这个闸门（见 Scheduler.runInference），一次性算完。
   */
  readonly minIntervalMs = 66

  readonly options: readonly TaskOptionSpec[] = [
    {
      key: 'points',
      label: '关键点',
      type: 'select',
      choices: [
        { value: 'all', label: '478 点 + 虹膜' },
        { value: 'contour', label: '仅轮廓' },
        { value: 'none', label: '不画' },
      ],
      default: 'all',
    },
    {
      key: 'mesh',
      label: '连线',
      type: 'select',
      choices: [
        { value: 'features', label: '五官' },
        { value: 'tess', label: '密集网格' },
        { value: 'off', label: '不画' },
      ],
      default: 'features',
    },
    {
      key: 'size',
      label: '点径',
      type: 'range',
      min: 0.5,
      max: 4,
      step: 0.5,
      default: 1.5,
      format: (v) => `${v}px`,
    },
    {
      key: 'blend',
      label: '表情',
      type: 'select',
      choices: [
        { value: 'off', label: '不显示' },
        { value: 'top3', label: '最强 3 项' },
      ],
      default: 'off',
    },
  ]

  private landmarker: FaceLandmarker | null = null
  private result: FaceLandmarkerResult | null = null
  private pointMode: PointMode = 'all'
  private meshMode: MeshMode = 'features'
  private blendMode: BlendMode = 'off'
  private pointSize = 1.5

  async init(report: (text: string, fraction: number) => void): Promise<void> {
    this.landmarker = await createVisionTask(MODEL, report, (fileset, modelAssetBuffer, delegate) =>
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer, delegate },
        runningMode: 'VIDEO',
        numFaces: 2,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: false,
      }),
    )
    // 关掉表情输出时也拿得到空数组，这里统一成开启，绘制时再决定画不画
    report('就绪', 1)
  }

  setOption(key: string, value: TaskOptionValue): void {
    switch (key) {
      case 'points':
        this.pointMode = value as PointMode
        break
      case 'mesh':
        this.meshMode = value as MeshMode
        break
      case 'blend':
        this.blendMode = value as BlendMode
        break
      case 'size':
        if (typeof value === 'number') this.pointSize = value
        break
    }
  }

  infer(ctx: FrameCtx): void {
    if (!this.landmarker) return
    // 静态图与视频统一走 detectForVideo：时间戳单调递增即可，静态图只会被调度器触发一次
    this.result = this.landmarker.detectForVideo(ctx.element, ctx.timestampMs)

    // 把结果放进帧内共享区，人脸识别任务据此做人脸对齐
    const faces = this.result.faceLandmarks ?? []
    ctx.shared.set(
      SHARED_FACE_LANDMARKS,
      faces.map((lm) => lm.map((p) => ({ x: p.x, y: p.y }))),
    )
    ctx.shared.set(SHARED_FACE_BOXES, faces.map((lm) => landmarkBox(lm)))
  }

  draw(o: Overlay): void {
    const faces = this.result?.faceLandmarks
    if (!faces || faces.length === 0) return

    faces.forEach((landmarks, faceIndex) => {
      const pts: Pt[] = landmarks.map((p) => ({ x: p.x, y: p.y }))

      if (this.meshMode === 'tess') {
        o.skeleton(pts, toPairs(FaceLandmarker.FACE_LANDMARKS_TESSELATION), {
          color: '#4c8dff',
          width: 0.5,
          alpha: 0.22,
        })
      } else if (this.meshMode === 'features') {
        const features = [
          FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
          FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
          FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
          FaceLandmarker.FACE_LANDMARKS_LIPS,
        ]
        for (const conn of features) {
          o.skeleton(pts, toPairs(conn), { color: '#4c8dff', width: 1.5, alpha: 0.85 })
        }
        // 虹膜单独一色，方便确认虹膜跟踪是否正常
        for (const conn of [
          FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
        ]) {
          o.skeleton(pts, toPairs(conn), { color: '#35c17b', width: 1.5, alpha: 0.9 })
        }
      }

      if (this.pointMode === 'all') {
        o.points(pts, {
          color: '#ffd166',
          radius: this.pointSize,
          outline: 'rgba(8,10,14,0.85)',
          alpha: 0.95,
        })
        const iris = [...LEFT_IRIS, ...RIGHT_IRIS].map((i) => pts[i]).filter(Boolean)
        o.points(iris, { color: '#35c17b', radius: this.pointSize + 0.5, outline: '#0a0c10' })
      } else if (this.pointMode === 'contour') {
        const oval = toPairs(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL).flat()
        o.points(
          [...new Set(oval)].map((i) => pts[i]).filter(Boolean),
          { color: '#ffd166', radius: this.pointSize + 0.5, outline: '#0a0c10' },
        )
      }

      if (this.blendMode === 'top3') {
        const shapes = this.result?.faceBlendshapes?.[faceIndex]?.categories ?? []
        const top = [...shapes].sort((a, b) => b.score - a.score).slice(0, 3)
        if (top.length > 0) {
          const box = landmarkBox(landmarks)
          o.label(
            box.x,
            box.y,
            top.map((c) => `${c.categoryName} ${(c.score * 100).toFixed(0)}%`).join('\n'),
            { color: '#35c17b', font: '11px ui-monospace, monospace' },
          )
        }
      }
    })
  }

  dispose(): void {
    this.landmarker?.close()
    this.landmarker = null
    this.result = null
  }
}

/** 由关键点算归一化外接框，外加一点边距 */
export function landmarkBox(landmarks: readonly { x: number; y: number }[]): Rect {
  let minX = 1
  let minY = 1
  let maxX = 0
  let maxY = 0
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  const padX = (maxX - minX) * 0.06
  const padY = (maxY - minY) * 0.06
  return {
    x: Math.max(0, minX - padX),
    y: Math.max(0, minY - padY),
    w: Math.min(1, maxX + padX) - Math.max(0, minX - padX),
    h: Math.min(1, maxY + padY) - Math.max(0, minY - padY),
  }
}

function toPairs(connections: readonly { start: number; end: number }[]): [number, number][] {
  return connections.map((c) => [c.start, c.end])
}
