import { ObjectDetector, type Detection } from '@mediapipe/tasks-vision'
import type { Overlay, Rect } from '../render/Overlay'
import { createVisionTask, type Delegate } from '../runtime/mediapipe'
import type { AssetSpec, FrameCtx, TaskOptionSpec, TaskOptionValue, VisionTask } from './types'

const MODEL: AssetSpec = {
  url: 'models/mediapipe/efficientdet_lite0.tflite',
  label: '物体检测模型',
  bytes: 6920000,
}

/** 一个检测结果，坐标已归一化 */
interface Box {
  rect: Rect
  label: string
  score: number
  color: string
}

/**
 * 按类别下标取色。
 *
 * 金角度 137.5° 在色环上铺得最均匀，任意前 N 个类别的颜色都不会撞得太近，
 * 比手写一张调色板省事，也不会因为类别数超过调色板长度而回绕重色。
 */
function colorFor(index: number, alpha = 1): string {
  const hue = (index * 137.508) % 360
  return `hsla(${hue.toFixed(0)}, 72%, 58%, ${alpha})`
}

/**
 * EfficientDet-Lite0 物体检测（COCO 80 类）。
 *
 * 该 `.tflite` 内嵌了 labelmap 元数据，MediaPipe 后处理（含 NMS）已经在图里做完，
 * 输出直接就是框 + 类别 + 置信度，不需要我们自己写解码。
 * 因此这里只做像素 -> 归一化坐标的换算。
 */
export class ObjectDetectTask implements VisionTask {
  readonly id = 'object-detect'
  readonly name = '物体检测'
  readonly stage = 'detect' as const
  readonly hint = 'EfficientDet-Lite0，COCO 80 类，输出框 + 类别 + 置信度'
  readonly assets = [MODEL]
  /**
   * 默认推理间隔。全开时几个任务会叠在同一帧里排队，帧时间直接变成它们的耗时之和，
   * 所以这里给每个任务定一个「够用就好」的重算频率：画面照 60fps 画，结果按各自的
   * 节奏更新。静态图不走这个闸门（见 Scheduler.runInference），一次性算完。
   */
  readonly minIntervalMs = 250

  readonly options: readonly TaskOptionSpec[] = [
    {
      key: 'threshold',
      label: '置信度',
      type: 'range',
      min: 0.1,
      max: 0.9,
      step: 0.05,
      default: 0.4,
      format: (v) => v.toFixed(2),
    },
    {
      key: 'max',
      label: '最多',
      type: 'range',
      min: 1,
      max: 10,
      step: 1,
      default: 5,
      format: (v) => `${v} 个`,
    },
    {
      key: 'labels',
      label: '标签',
      type: 'select',
      choices: [
        { value: 'name', label: '类别名' },
        { value: 'score', label: '类别 + 置信度' },
        { value: 'off', label: '只画框' },
      ],
      default: 'score',
    },
  ]

  private detector: ObjectDetector | null = null
  /** setOptions 要带上完整配置，模型字节与 delegate 得留着 */
  private modelAssetBuffer: Uint8Array | null = null
  private delegate: Delegate = 'CPU'

  private boxes: Box[] = []
  private threshold = 0.4
  private maxResults = 5
  private labelMode: 'name' | 'score' | 'off' = 'score'

  async init(report: (text: string, fraction: number) => void): Promise<void> {
    this.detector = await createVisionTask(MODEL, report, (fileset, modelAssetBuffer, delegate) => {
      this.modelAssetBuffer = modelAssetBuffer
      this.delegate = delegate
      return ObjectDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer, delegate },
        runningMode: 'VIDEO',
        scoreThreshold: this.threshold,
        maxResults: this.maxResults,
      })
    })
    report('就绪', 1)
  }

  setOption(key: string, value: TaskOptionValue): void {
    if (key === 'threshold' && typeof value === 'number') {
      this.threshold = value
      this.reconfigure()
    }
    if (key === 'max' && typeof value === 'number') {
      this.maxResults = value
      this.reconfigure()
    }
    if (key === 'labels' && (value === 'name' || value === 'score' || value === 'off')) {
      this.labelMode = value
    }
  }

  /** 阈值/上限是构图期参数，改写运行中的检测器即可，不必重建模型 */
  private reconfigure(): void {
    if (!this.detector || !this.modelAssetBuffer) return
    this.detector.setOptions({
      baseOptions: { modelAssetBuffer: this.modelAssetBuffer, delegate: this.delegate },
      runningMode: 'VIDEO',
      scoreThreshold: this.threshold,
      maxResults: this.maxResults,
    })
  }

  infer(ctx: FrameCtx): void {
    if (!this.detector) return
    const res = this.detector.detectForVideo(ctx.element, ctx.timestampMs)
    // 同 faceDetect：框是**推理输入**那份画面的像素坐标，得除以 elementW/H 而不是源尺寸
    const w = ctx.elementW || 1
    const h = ctx.elementH || 1

    this.boxes = res.detections.slice(0, this.maxResults).flatMap((det: Detection) => {
      const b = det.boundingBox
      if (!b) return []
      const top = det.categories[0]
      const index = top?.index ?? 0
      return [
        {
          rect: { x: b.originX / w, y: b.originY / h, w: b.width / w, h: b.height / h },
          label: top?.categoryName || top?.displayName || `#${index}`,
          score: top?.score ?? 0,
          color: colorFor(index),
        },
      ]
    })
  }

  draw(o: Overlay): void {
    for (const box of this.boxes) {
      o.rect(box.rect, { color: box.color, width: 2 })
      if (this.labelMode === 'off') continue
      const text =
        this.labelMode === 'name' ? box.label : `${box.label} ${(box.score * 100).toFixed(0)}%`
      o.label(box.rect.x, box.rect.y, text, {
        foreground: '#0a0c10',
        background: box.color,
        font: '600 12px system-ui, sans-serif',
      })
    }
  }

  dispose(): void {
    this.detector?.close()
    this.detector = null
    this.modelAssetBuffer = null
    this.boxes = []
  }
}
