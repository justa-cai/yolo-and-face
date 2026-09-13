import { ImageClassifier, type ImageClassifierResult } from '@mediapipe/tasks-vision'
import type { Overlay, PanelRow } from '../render/Overlay'
import { createVisionTask, type Delegate } from '../runtime/mediapipe'
import type { AssetSpec, FrameCtx, TaskOptionSpec, TaskOptionValue, VisionTask } from './types'

const MODEL: AssetSpec = {
  url: 'models/mediapipe/efficientnet_lite0.tflite',
  label: '图像分类模型',
  bytes: 17720000,
}

/**
 * EfficientNet-Lite0 整帧分类（ImageNet 1000 类）。
 *
 * 和「物体检测」的区别：检测给出「画面里有哪些东西、各在哪」，分类只回答
 * 「整张图最像什么」，两者互补——检测框不出来（物体太小/遮挡）时，分类
 * 往往还能给出一个有意义的判断。
 */
export class ClassifyTask implements VisionTask {
  readonly id = 'classify'
  readonly name = '图像分类'
  readonly stage = 'classify' as const
  readonly hint = 'EfficientNet-Lite0，ImageNet 1000 类，输出整帧 Top-N 排行榜'
  readonly assets = [MODEL]
  /**
   * 默认推理间隔。全开时几个任务会叠在同一帧里排队，帧时间直接变成它们的耗时之和，
   * 所以这里给每个任务定一个「够用就好」的重算频率：画面照 60fps 画，结果按各自的
   * 节奏更新。静态图不走这个闸门（见 Scheduler.runInference），一次性算完。
   */
  readonly minIntervalMs = 250

  readonly options: readonly TaskOptionSpec[] = [
    {
      key: 'max',
      label: 'Top-N',
      type: 'range',
      min: 1,
      max: 8,
      step: 1,
      default: 5,
      format: (v) => `前 ${v} 类`,
    },
    {
      key: 'threshold',
      label: '置信下限',
      type: 'range',
      min: 0,
      max: 0.9,
      step: 0.05,
      default: 0.05,
      format: (v) => v.toFixed(2),
    },
    {
      key: 'position',
      label: '位置',
      type: 'select',
      choices: [
        { value: 'tl', label: '左上' },
        { value: 'tr', label: '右上' },
        { value: 'bl', label: '左下' },
        { value: 'br', label: '右下' },
      ],
      default: 'tl',
    },
  ]

  private classifier: ImageClassifier | null = null
  /** setOptions 要带上完整配置，模型字节与 delegate 得留着 */
  private modelAssetBuffer: Uint8Array | null = null
  private delegate: Delegate = 'CPU'
  private result: ImageClassifierResult | null = null

  private maxResults = 5
  private threshold = 0.05
  private corner: 'tl' | 'tr' | 'bl' | 'br' = 'tl'

  async init(report: (text: string, fraction: number) => void): Promise<void> {
    this.classifier = await createVisionTask(MODEL, report, (fileset, modelAssetBuffer, delegate) => {
      this.modelAssetBuffer = modelAssetBuffer
      this.delegate = delegate
      return ImageClassifier.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer, delegate },
        runningMode: 'VIDEO',
        maxResults: this.maxResults,
        scoreThreshold: this.threshold,
      })
    })
    report('就绪', 1)
  }

  setOption(key: string, value: TaskOptionValue): void {
    if (key === 'max' && typeof value === 'number') {
      this.maxResults = value
      this.reconfigure()
    }
    if (key === 'threshold' && typeof value === 'number') {
      this.threshold = value
      this.reconfigure()
    }
    if (key === 'position' && (value === 'tl' || value === 'tr' || value === 'bl' || value === 'br')) {
      this.corner = value
    }
  }

  /** 两项都是构图期参数，改写运行中的分类器即可，不必重建模型 */
  private reconfigure(): void {
    if (!this.classifier || !this.modelAssetBuffer) return
    this.classifier.setOptions({
      baseOptions: { modelAssetBuffer: this.modelAssetBuffer, delegate: this.delegate },
      runningMode: 'VIDEO',
      maxResults: this.maxResults,
      scoreThreshold: this.threshold,
    })
  }

  infer(ctx: FrameCtx): void {
    if (!this.classifier) return
    this.result = this.classifier.classifyForVideo(ctx.element, ctx.timestampMs)
  }

  draw(o: Overlay): void {
    const rows = this.topRows()
    if (rows.length === 0) return

    // 面板位置在归一化空间里算：内容尺寸不为 0 时才能换算像素估值
    const m = 0.012
    const left = this.corner === 'tl' || this.corner === 'bl'
    const top = this.corner === 'tl' || this.corner === 'tr'
    const wNorm = o.contentWidth > 0 ? this.estimateWidth(rows) / o.contentWidth : 0
    const hNorm = o.contentHeight > 0 ? this.estimateHeight(rows) / o.contentHeight : 0

    o.panel(left ? m : 1 - m - wNorm, top ? m : 1 - m - hNorm, rows, {
      title: `图像分类 Top-${rows.length}`,
      max: rows[0].value,
      barColor: '#35c17b',
    })
  }

  private topRows(): PanelRow[] {
    const head = this.result?.classifications[0]
    if (!head) return []
    return head.categories
      .filter((c) => (c.score ?? 0) >= this.threshold)
      .slice(0, this.maxResults)
      .map((c) => ({
        text: `${(c.score * 100).toFixed(0)}%  ${c.categoryName || c.displayName || `#${c.index}`}`,
        value: c.score,
      }))
  }

  /**
   * 只是为了让「右下 / 左下」的对齐锚点大致落在面板左上角。
   * 面板本身还会自己做一次越界翻转，所以这里估偏几个像素不影响可读性。
   */
  private estimateWidth(rows: readonly PanelRow[]): number {
    const longest = rows.reduce((m, r) => Math.max(m, r.text.length), 0)
    const titleLen = `图像分类 Top-${rows.length}`.length
    return Math.max(longest, titleLen) * 7 + 104
  }

  private estimateHeight(rows: readonly PanelRow[]): number {
    return rows.length * 18 + 34
  }

  dispose(): void {
    this.classifier?.close()
    this.classifier = null
    this.modelAssetBuffer = null
    this.result = null
  }
}
