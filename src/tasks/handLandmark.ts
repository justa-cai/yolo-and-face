import { HandLandmarker, type HandLandmarkerResult } from '@mediapipe/tasks-vision'
import type { Overlay, Pt, Rect } from '../render/Overlay'
import { createVisionTask } from '../runtime/mediapipe'
import type { AssetSpec, FrameCtx, TaskOptionSpec, TaskOptionValue, VisionTask } from './types'

/** 供掌纹识别任务取用：最近一帧每只手的 21 个归一化关键点 */
export const SHARED_HAND_LANDMARKS = 'hand.landmarks'

/** 每只手的关键点数（MediaPipe 固定 21） */
export const HAND_POINT_COUNT = 21

const MODEL: AssetSpec = {
  url: 'models/mediapipe/hand_landmarker.task',
  label: '手部关键点模型',
  bytes: 7819105,
}

type PointMode = 'all' | 'palm' | 'none'

export class HandLandmarkTask implements VisionTask {
  readonly id = 'hand-landmark'
  readonly name = '手部关键点'
  readonly stage = 'landmark' as const
  readonly hint = '21 个手部关键点（含指尖与掌指关节）。掌纹识别靠它定位掌心'
  readonly assets = [MODEL]
  /** 手部检测比人脸重，降到 12Hz 左右够用；绘制照常每帧进行 */
  readonly minIntervalMs = 80

  readonly options: readonly TaskOptionSpec[] = [
    {
      key: 'hands',
      label: '手数',
      type: 'range',
      min: 1,
      max: 2,
      step: 1,
      default: 2,
      format: (v) => `${v} 只`,
    },
    {
      key: 'points',
      label: '关键点',
      type: 'select',
      choices: [
        { value: 'all', label: '21 点' },
        { value: 'palm', label: '仅定位点' },
        { value: 'none', label: '不画' },
      ],
      default: 'all',
    },
    {
      key: 'skeleton',
      label: '连线',
      type: 'select',
      choices: [
        { value: 'on', label: '画' },
        { value: 'off', label: '不画' },
      ],
      default: 'on',
    },
  ]

  private landmarker: HandLandmarker | null = null
  private result: HandLandmarkerResult | null = null
  private numHands = 2
  private pointMode: PointMode = 'all'
  private showSkeleton = true
  private delegate: 'GPU' | 'CPU' = 'CPU'
  private modelAssetBuffer: Uint8Array | null = null

  async init(report: (text: string, fraction: number) => void): Promise<void> {
    this.landmarker = await createVisionTask(MODEL, report, (fileset, modelAssetBuffer, delegate) => {
      this.delegate = delegate
      this.modelAssetBuffer = modelAssetBuffer
      return HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer, delegate },
        runningMode: 'VIDEO',
        numHands: this.numHands,
      })
    })
    report('就绪', 1)
  }

  setOption(key: string, value: TaskOptionValue): void {
    switch (key) {
      case 'hands':
        if (typeof value === 'number') {
          this.numHands = value
          this.reconfigure()
        }
        break
      case 'points':
        this.pointMode = value as PointMode
        break
      case 'skeleton':
        this.showSkeleton = value === 'on'
        break
    }
  }

  /** 手数是构图期参数，改它得重建推理图；模型 buffer 必须一并带上 */
  private reconfigure(): void {
    if (!this.landmarker || !this.modelAssetBuffer) return
    this.landmarker.setOptions({
      baseOptions: { modelAssetBuffer: this.modelAssetBuffer, delegate: this.delegate },
      runningMode: 'VIDEO',
      numHands: this.numHands,
    })
  }

  infer(ctx: FrameCtx): void {
    if (!this.landmarker) return
    this.result = this.landmarker.detectForVideo(ctx.element, ctx.timestampMs)

    const hands = this.result.landmarks ?? []
    ctx.shared.set(
      SHARED_HAND_LANDMARKS,
      hands.map((lm) => lm.map((p) => ({ x: p.x, y: p.y }))),
    )
  }

  draw(o: Overlay): void {
    const hands = this.result?.landmarks
    if (!hands || hands.length === 0) return

    for (const landmarks of hands) {
      const pts: Pt[] = landmarks.map((p) => ({ x: p.x, y: p.y }))

      if (this.showSkeleton) {
        o.skeleton(pts, HAND_CONNECTIONS, { color: '#4c8dff', width: 2, alpha: 0.9 })
      }

      if (this.pointMode === 'all') {
        o.points(pts, {
          color: '#ffd166',
          radius: 2.5,
          outline: 'rgba(8,10,14,0.85)',
          alpha: 0.95,
        })
      } else if (this.pointMode === 'palm') {
        // 只画掌纹 ROI 真正用到的那几个点：两个指间谷、手腕、以及组成谷的四个掌指关节
        const idx = [0, 5, 9, 13, 17]
        o.points(
          idx.map((i) => pts[i]).filter(Boolean),
          { color: '#35c17b', radius: 3.5, outline: '#0a0c10' },
        )
      }
    }
  }

  dispose(): void {
    this.landmarker?.close()
    this.landmarker = null
    this.modelAssetBuffer = null
    this.result = null
  }
}

/** MediaPipe 的 21 点骨架连线（下标见 HandLandmarker.HAND_CONNECTIONS） */
const HAND_CONNECTIONS: [number, number][] = [
  // 拇指
  [0, 1], [1, 2], [2, 3], [3, 4],
  // 食指
  [0, 5], [5, 6], [6, 7], [7, 8],
  // 中指
  [5, 9], [9, 10], [10, 11], [11, 12],
  // 无名指
  [9, 13], [13, 14], [14, 15], [15, 16],
  // 小指
  [13, 17], [17, 18], [18, 19], [19, 20],
  // 掌根
  [0, 17],
]

/** 由关键点算归一化外接框，外加一点边距（与人脸那套保持一致的手感） */
export function handBox(landmarks: readonly { x: number; y: number }[]): Rect {
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
