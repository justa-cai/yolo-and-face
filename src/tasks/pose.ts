import { PoseLandmarker, type PoseLandmarkerResult } from '@mediapipe/tasks-vision'
import type { Overlay, Pt } from '../render/Overlay'
import { createVisionTask } from '../runtime/mediapipe'
import type { AssetSpec, FrameCtx, TaskOptionSpec, TaskOptionValue, VisionTask } from './types'

type ModelKey = 'lite' | 'full'

const MODELS: Record<ModelKey, AssetSpec> = {
  lite: { url: 'models/mediapipe/pose_landmarker_lite.task', label: '姿态模型 (Lite)' },
  full: { url: 'models/mediapipe/pose_landmarker_full.task', label: '姿态模型 (Full)' },
}

/** 关键点下标，用于给骨架分段上色 */
const LEFT_SIDE = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
const RIGHT_SIDE = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

/**
 * 单人姿态估计（33 点）。
 *
 * 只做单人：上层明确不需要多人，MediaPipe 的 PoseLandmarker 在单人场景下
 * 速度与稳定性都明显更好，也避开了 YOLO-Pose 那套自写 letterbox / NMS 的负担。
 * 将来若要多人，再单独接 RTMPose（Apache-2.0）而不是 YOLO-Pose（AGPL）。
 */
export class PoseTask implements VisionTask {
  readonly id = 'pose'
  readonly name = '姿态 33 点'
  readonly stage = 'landmark' as const
  readonly hint = '单人 33 个姿态关键点，含四肢与面部轮廓点'
  /**
   * 默认推理间隔。全开时几个任务会叠在同一帧里排队，帧时间直接变成它们的耗时之和，
   * 所以这里给每个任务定一个「够用就好」的重算频率：画面照 60fps 画，结果按各自的
   * 节奏更新。静态图不走这个闸门（见 Scheduler.runInference），一次性算完。
   */
  readonly minIntervalMs = 66

  readonly options: readonly TaskOptionSpec[] = [
    {
      key: 'model',
      label: '模型',
      type: 'select',
      choices: [
        { value: 'lite', label: 'Lite（快）' },
        { value: 'full', label: 'Full（准）' },
      ],
      default: 'lite',
    },
    {
      key: 'threshold',
      label: '置信度',
      type: 'range',
      min: 0.1,
      max: 0.95,
      step: 0.05,
      default: 0.5,
      format: (v) => v.toFixed(2),
    },
    {
      key: 'size',
      label: '点径',
      type: 'range',
      min: 1,
      max: 8,
      step: 0.5,
      default: 4,
      format: (v) => `${v}px`,
    },
    {
      key: 'depth',
      label: '深度',
      type: 'select',
      choices: [
        { value: 'on', label: '按远近变化' },
        { value: 'off', label: '固定大小' },
      ],
      default: 'on',
    },
  ]

  private landmarker: PoseLandmarker | null = null
  private result: PoseLandmarkerResult | null = null
  private report: ((text: string, fraction: number) => void) | null = null

  private modelKey: ModelKey = 'lite'
  private threshold = 0.5
  private pointSize = 4
  private depthCue = true

  get assets(): readonly AssetSpec[] {
    return [MODELS[this.modelKey]]
  }

  async init(report: (text: string, fraction: number) => void): Promise<void> {
    this.report = report
    await this.load()
  }

  private async load(): Promise<void> {
    const report = this.report ?? (() => {})
    this.landmarker = await createVisionTask(MODELS[this.modelKey], report, (fileset, buf, delegate) =>
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: buf, delegate },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: this.threshold,
        minPosePresenceConfidence: this.threshold,
        minTrackingConfidence: this.threshold,
      }),
    )
    report('就绪', 1)
  }

  setOption(key: string, value: TaskOptionValue): void {
    if (key === 'model' && (value === 'lite' || value === 'full')) {
      if (this.modelKey === value) return
      this.modelKey = value
      // 换模型只能重建：MediaPipe 不支持换掉已装载的模型 buffer
      this.dispose()
      void this.load().catch((err) => console.error('[pose] 切换模型失败', err))
      return
    }
    if (key === 'threshold' && typeof value === 'number') {
      this.threshold = value
      this.reconfigure()
    }
    if (key === 'size' && typeof value === 'number') this.pointSize = value
    if (key === 'depth') this.depthCue = value === 'on'
  }

  /** 三个置信度阈值是构图期参数，改写运行中的实例即可，不必重建模型 */
  private reconfigure(): void {
    if (!this.landmarker) return
    this.landmarker.setOptions({
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: this.threshold,
      minPosePresenceConfidence: this.threshold,
      minTrackingConfidence: this.threshold,
    })
  }

  infer(ctx: FrameCtx): void {
    if (!this.landmarker) return
    this.result = this.landmarker.detectForVideo(ctx.element, ctx.timestampMs)
  }

  draw(o: Overlay): void {
    const poses = this.result?.landmarks
    if (!poses || poses.length === 0) return

    for (const landmarks of poses) {
      const pts: Pt[] = landmarks.map((p) => ({ x: p.x, y: p.y }))
      const vis = landmarks.map((p) => p.visibility ?? 1)

      o.skeleton(pts, toPairs(PoseLandmarker.POSE_CONNECTIONS), {
        color: '#4c8dff',
        width: 3,
        alpha: 0.9,
      })

      // 躯干单列出来加粗，避免四肢挡住主体
      const torso: [number, number][] = [
        [11, 12],
        [11, 23],
        [12, 24],
        [23, 24],
      ]
      o.skeleton(pts, torso, { color: '#e7eaf0', width: 3, alpha: 0.75 })

      // 左右肢分色，方便一眼看出交叉/遮挡
      o.points(pick(pts, LEFT_SIDE, vis), {
        color: '#35c17b',
        radius: this.pointSize,
        outline: '#0a0c10',
      })
      o.points(pick(pts, RIGHT_SIDE, vis), {
        color: '#e0a13a',
        radius: this.pointSize,
        outline: '#0a0c10',
      })

      // 面部与其余点（0-10 是面部轮廓与肩中点）
      const head = pts
        .map((p, i) => ({ p, i }))
        .filter(({ i }) => i < 11 && (vis[i] ?? 1) > 0.5)
        .map(({ p }) => p)
      o.points(head, { color: '#8d95a5', radius: this.pointSize * 0.7, outline: '#0a0c10' })

      // 用 z 做远近暗示：越远点越小越淡（z 是相对髋部中心的深度，单位约等于躯干长度）
      if (this.depthCue) {
        const far = landmarks
          .map((p, i) => ({ p, i }))
          .filter(({ p, i }) => (p.z ?? 0) > 0.15 && (vis[i] ?? 1) > 0.5)
          .map(({ p }) => ({ x: p.x, y: p.y }))
        if (far.length > 0) {
          o.points(far, {
            color: 'rgba(76,141,255,0.35)',
            radius: Math.max(1, this.pointSize - 1.5),
          })
        }
      }
    }
  }

  dispose(): void {
    this.landmarker?.close()
    this.landmarker = null
    this.result = null
  }
}

function pick(pts: readonly Pt[], indices: readonly number[], vis: readonly number[]): Pt[] {
  const out: Pt[] = []
  for (const i of indices) {
    const p = pts[i]
    if (p && (vis[i] ?? 1) > 0.5) out.push(p)
  }
  return out
}

function toPairs(connections: readonly { start: number; end: number }[]): [number, number][] {
  return connections.map((c) => [c.start, c.end])
}
