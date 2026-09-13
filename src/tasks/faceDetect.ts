import { FaceDetector, type Detection } from '@mediapipe/tasks-vision'
import type { Overlay, Rect } from '../render/Overlay'
import { createVisionTask, type Delegate } from '../runtime/mediapipe'
import type { AssetSpec, FrameCtx, TaskOptionSpec, TaskOptionValue, VisionTask } from './types'

const MODEL: AssetSpec = {
  url: 'models/mediapipe/blaze_face_short_range.tflite',
  label: '人脸检测模型',
  bytes: 232000,
}

/** 归一化后的人脸框 + 置信度 */
interface FaceBox {
  rect: Rect
  score: number
  keypoints: { x: number; y: number }[]
}

/**
 * BlazeFace 短距人脸检测。
 *
 * 只出人脸框和 6 个关键点，比 478 点关键点模型便宜得多，适合"只要知道
 * 画面里有没有脸、在哪"的场景。需要五官细节时用「人脸关键点」任务。
 */
export class FaceDetectTask implements VisionTask {
  readonly id = 'face-detect'
  readonly name = '人脸检测'
  readonly stage = 'detect' as const
  readonly hint = 'BlazeFace 短距模型，输出人脸框与 6 个关键点，速度快'
  readonly assets = [MODEL]
  /**
   * 默认推理间隔。全开时几个任务会叠在同一帧里排队，帧时间直接变成它们的耗时之和，
   * 所以这里给每个任务定一个「够用就好」的重算频率：画面照 60fps 画，结果按各自的
   * 节奏更新。静态图不走这个闸门（见 Scheduler.runInference），一次性算完。
   */
  readonly minIntervalMs = 66

  readonly options: readonly TaskOptionSpec[] = [
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
      key: 'faces',
      label: '最多',
      type: 'range',
      min: 1,
      max: 8,
      step: 1,
      default: 4,
      format: (v) => `${v} 张`,
    },
    {
      key: 'keypoints',
      label: '关键点',
      type: 'select',
      choices: [
        { value: 'on', label: '显示 6 点' },
        { value: 'off', label: '不显示' },
      ],
      default: 'on',
    },
  ]

  private detector: FaceDetector | null = null
  /** 改写阈值时要带上完整配置，所以模型字节和 delegate 都得留着 */
  private modelAssetBuffer: Uint8Array | null = null
  private delegate: Delegate = 'CPU'

  private boxes: FaceBox[] = []
  private threshold = 0.5
  private numFaces = 4
  private showKeypoints = true

  async init(report: (text: string, fraction: number) => void): Promise<void> {
    this.detector = await createVisionTask(MODEL, report, (fileset, modelAssetBuffer, delegate) => {
      this.modelAssetBuffer = modelAssetBuffer
      this.delegate = delegate
      return FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer, delegate },
        runningMode: 'VIDEO',
        minDetectionConfidence: this.threshold,
      })
    })
    report('就绪', 1)
  }

  setOption(key: string, value: TaskOptionValue): void {
    if (key === 'threshold' && typeof value === 'number') {
      this.threshold = value
      this.reconfigure()
    }
    if (key === 'faces' && typeof value === 'number') this.numFaces = value
    if (key === 'keypoints') this.showKeypoints = value === 'on'
  }

  /** 阈值是构图期参数，只能改写运行中的检测器；模型 buffer 必须一并带上 */
  private reconfigure(): void {
    if (!this.detector || !this.modelAssetBuffer) return
    this.detector.setOptions({
      baseOptions: { modelAssetBuffer: this.modelAssetBuffer, delegate: this.delegate },
      runningMode: 'VIDEO',
      minDetectionConfidence: this.threshold,
    })
  }

  infer(ctx: FrameCtx): void {
    if (!this.detector) return
    const res = this.detector.detectForVideo(ctx.element, ctx.timestampMs)
    // 必须除以**推理输入**的尺寸，不是源画面：调度器会把画面降采样后再交给 ctx.element，
    // BlazeFace 的 boundingBox 用的是那份的像素坐标。除以源尺寸会得到一个被压缩到
    // 左上角的框（缩放比 0.5 时框只有一半大）。
    const w = ctx.elementW || 1
    const h = ctx.elementH || 1

    // BlazeFace 的 boundingBox 是输入帧的像素坐标，在这里一次性换算成归一化坐标，
    // 绘制阶段就不需要知道帧尺寸了
    this.boxes = res.detections.slice(0, this.numFaces).flatMap((det: Detection) => {
      const b = det.boundingBox
      if (!b) return []
      return [
        {
          rect: { x: b.originX / w, y: b.originY / h, w: b.width / w, h: b.height / h },
          score: det.categories[0]?.score ?? 0,
          keypoints: det.keypoints.map((k) => ({ x: k.x, y: k.y })),
        },
      ]
    })
  }

  draw(o: Overlay): void {
    for (const face of this.boxes) {
      o.rect(face.rect, { color: '#4c8dff', width: 2 })
      o.label(face.rect.x, face.rect.y, `face ${(face.score * 100).toFixed(0)}%`)
      if (this.showKeypoints) {
        o.points(face.keypoints, { color: '#35c17b', radius: 2.5, outline: '#0a0c10' })
      }
    }
  }

  dispose(): void {
    this.detector?.close()
    this.detector = null
    this.modelAssetBuffer = null
    this.boxes = []
  }
}
