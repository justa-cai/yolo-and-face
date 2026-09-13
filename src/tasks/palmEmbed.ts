import type { OrtModule } from '../runtime/ort'
import { createOrtSession, loadOrt } from '../runtime/ort'
import type { Overlay, Pt, Rect } from '../render/Overlay'
import { PALM_SIZE, palmRoiQuad, palmRoiToTensor } from '../palm/roi'
import { findBackbone, PALM_BACKBONES, type PalmBackbone } from '../palm/backbone'
import { l2Normalize, matchEmbedding, type EmbeddingEntry, type EmbeddingMatch } from '../biometric/match'
import type { EmbeddingGallery } from '../biometric/gallery'
import { handBox, SHARED_HAND_LANDMARKS } from './handLandmark'
import type { AssetSpec, BiometricTask, FrameCtx, TaskOptionSpec, TaskOptionValue } from './types'

/**
 * 默认判定阈值。
 *
 * 实测标定的结果（6 张公开手掌照片、掌心 ROI 已按 src/palm/roi.ts 归一化，
 * 明细见 tmp/palm_calib.json，脚本 tmp/palm_calib.py 可复跑）：
 *
 * | 骨干 | 同手相似度最低 | 异手最高 | 该骨干的可用区间 |
 * |---|---|---|---|
 * | DINOv2-Small | 0.837 | 0.738 | 0.738 – 0.837 |
 * | MobileNetV3 | 0.867 | 0.703 | 0.703 – 0.867 |
 *
 * 「异手最高」已经把 palm01/palm10 那一对**排除掉了**：那两张是同一来源、疑似同一只手，
 * 互相能到 0.94/0.92。它们到底是不是同一个人的手，我判断不了——如果是，
 * 那 0.94 正是「同手」该有的分数；如果不是，就说明这套特征会误判，而这 6 个样本
 * 里再没有第三对这么像的可以用来分辨。这条不确定性必须跟着这个功能一起说清楚。
 *
 * 「同手」这一侧是把原图旋转 ±8°、缩放 0.9/1.08、亮度 ±8% 之后重跑关键点和 ROI 得到的，
 * 比真实摄像头的姿态变化温和，**实际使用中的漏识会明显比这张表更严重**。
 *
 * 0.79 落在两个骨干可用区间的交集 (0.738, 0.837) 中间，两边都留出约 0.05 的余量。
 * 换骨干不用改这个值；但换了**模型**就必须重新标定。
 */
export const PALM_THRESHOLD = 0.79

/** 一帧里一只手的结果 */
export interface PalmIdentification {  rect: Rect
  /** ROI 的四个角（归一化坐标），用来把这块区域画出来 */
  quad: Pt[]
  /** 与掌纹库里最像的那条的余弦相似度；库为空时为 null */
  score: number | null
  /** 命中的条目；未过阈值时为 null */
  hit: EmbeddingMatch | null
  /** 特征向量，注册掌纹时要用 */
  embedding: Float32Array
}

/**
 * 掌纹识别 1:N（实验性）。
 *
 * ⚠️ **这不是一个可用的身份认证方案**，只是一个「通用骨干 + few-shot 匹配」的探索：
 * 掌纹领域没有许可宽松的预训练模型，所以这里用的是通用视觉骨干，判别力没有经过
 * 任何公开数据集评估，阈值是拿 6 张自己的手掌照片实测定的。演示用途，别拿去当真身份核验。
 *
 * ## 它到底在比什么
 *
 * 说得更直白一点：**它比的不是掌纹脊线，是掌心的整体外观**。
 * 实测这几张照片里掌宽大约 320 像素，成人掌心横向有 150–200 条脊线，
 * 也就是每条约 2 像素——本来就在采样极限上，再压到 224×224 更是彻底糊掉。
 * 所以这套特征主要抓的是肤色、手型、掌纹走向和光照，而不是能唯一标识个人的脊线细节。
 * 两个人手掌外观相近时它**会**误判，这一点在 6 个样本上无法评估，别抱幻想。
 *
 * 流程本身是完整的、也是这类系统的标准骨架：
 * 手部 21 点 → 掌心 ROI 归一化 → 骨干抽特征 → 余弦比对。
 * 换成真正的掌纹模型（要有训练数据）就能直接替换骨干那一步。
 */
export class PalmRecognizeTask implements BiometricTask {
  readonly id = 'palm-recognize'
  readonly name = '掌纹识别 1:N（实验性）'
  readonly stage = 'classify' as const
  readonly hint =
    '实验性功能：用通用视觉骨干抽掌心特征做 1:N 比对，判别力未经公开数据集评估，不可用于真实身份认证'
  readonly dependsOn = ['hand-landmark'] as const
  /** 换骨干等于换模型，必须重建会话 */
  readonly reinitOn = ['backbone'] as const
  readonly minIntervalMs = 150

  readonly options: readonly TaskOptionSpec[] = [
    {
      key: 'backbone',
      label: '骨干',
      type: 'select',
      choices: PALM_BACKBONES.map((b) => ({ value: b.id, label: b.label })),
      default: 'dinov2',
    },
    {
      key: 'threshold',
      label: '判定阈值',
      type: 'range',
      min: 0.5,
      max: 0.999,
      step: 0.001,
      default: PALM_THRESHOLD,
      format: (v) => v.toFixed(3),
    },
    {
      key: 'topk',
      label: '候选数',
      type: 'range',
      min: 1,
      max: 3,
      step: 1,
      default: 1,
      format: (v) => `前 ${v} 名`,
    },
    {
      key: 'roi',
      label: 'ROI',
      type: 'select',
      choices: [
        { value: 'on', label: '画出来' },
        { value: 'off', label: '不画' },
      ],
      default: 'on',
    },
    {
      key: 'unknown',
      label: '未命中',
      type: 'select',
      choices: [
        { value: 'label', label: '显示相似度' },
        { value: 'hide', label: '不画框' },
      ],
      default: 'label',
    },
  ]

  private ort: OrtModule | null = null
  private session: import('onnxruntime-web').InferenceSession | null = null
  private inputName = ''
  private outputName = ''
  private backend = ''
  private backbone: PalmBackbone = findBackbone('dinov2')

  private readonly gallery: EmbeddingGallery
  private entries: EmbeddingEntry[] = []
  private results: PalmIdentification[] = []

  private threshold = PALM_THRESHOLD
  private topK = 1
  private showUnknown = true
  private showRoi = true

  constructor(gallery: EmbeddingGallery) {
    this.gallery = gallery
  }

  get assets(): readonly AssetSpec[] {
    return [this.backbone.asset]
  }

  async init(report: (text: string, fraction: number) => void): Promise<void> {
    const { session, backend } = await createOrtSession(this.backbone.asset, report)
    this.session = session
    this.backend = backend
    this.inputName = session.inputNames[0]
    this.outputName = session.outputNames[0]
    this.ort = await loadOrt()
    await this.reloadGallery()
    report('就绪', 1)
  }

  /** 注册/删除掌纹后调用，让下一帧用新的库 */
  async reloadGallery(): Promise<void> {
    try {
      this.entries = await this.gallery.list()
    } catch (err) {
      console.error('[palm-recognize] 读取掌纹库失败', err)
      this.entries = []
    }
  }

  get backendLabel(): string {
    return this.backend === 'webgpu' ? 'ONNX WebGPU' : 'ONNX WASM'
  }

  /** 当前骨干的展示名，侧栏用来提示「换骨干会清空比对结果」 */
  get backboneLabel(): string {
    return this.backbone.label
  }

  get backboneId(): string {
    return this.backbone.id
  }

  /**
   * 库里有向量维度对不上当前骨干时的提示，没有则返回 null。
   *
   * 换骨干会换特征空间：DINOv2 是 384 维、MobileNetV3 是 1024 维，同一个库里
   * 两种向量混着放，比对时那些对不上的条目会被静默跳过——不提示的话，
   * 用户只会看到「明明注册过却一直显示未知」，完全不知道为什么。
   */
  get galleryWarning(): string | null {
    const total = this.entries.length
    if (total === 0) return null
    const usable = this.entries.filter((e) => e.embedding.length === this.featureDim()).length
    if (usable === total) return null
    return `库里有 ${total - usable} 条特征与当前骨干「${this.backbone.label}」维度不符，` +
      `这些条目不会参与比对；要按当前骨干重新注册`
  }

  /** 当前骨干吐出的特征维度。优先问模型自己，问不到再用骨干声明的值。 */
  private featureDim(): number {
    // outputMetadata 在类型上是 tensor / 非 tensor 的联合，非 tensor 没有 shape
    const meta = this.session?.outputMetadata?.[0]
    const shape = meta && 'shape' in meta ? meta.shape : null
    if (shape && shape.length > 0 && shape.every((d) => typeof d === 'number')) {
      return this.backbone.featureLength(shape as number[])
    }
    return this.backbone.dimension
  }

  /** 最近一帧里第一只手的特征，供给「注册当前掌纹」用 */
  captureEmbedding(): Float32Array | null {
    return this.results[0]?.embedding ?? null
  }

  setOption(key: string, value: TaskOptionValue): void {
    if (key === 'backbone' && typeof value === 'string') this.backbone = findBackbone(value)
    if (key === 'threshold' && typeof value === 'number') this.threshold = value
    if (key === 'topk' && typeof value === 'number') this.topK = value
    if (key === 'roi') this.showRoi = value === 'on'
    if (key === 'unknown') this.showUnknown = value === 'label'
  }

  async infer(ctx: FrameCtx): Promise<void> {
    const session = this.session
    if (!session || !this.ort) return

    const hands = ctx.shared.get(SHARED_HAND_LANDMARKS) as Pt[][] | undefined
    if (!hands || hands.length === 0) {
      this.results = []
      return
    }

    const out: PalmIdentification[] = []
    for (const landmarks of hands) {
      const quad = palmRoiQuad(landmarks, ctx.width, ctx.height)
      if (!quad) continue

      // 与掌纹识别一样刻意用原始分辨率的画面：ROI 只有 224×224，
      // 源画面越清楚，重采样后越接近骨干训练时的成像条件
      const tensor = palmRoiToTensor(ctx.source.element, landmarks, ctx.width, ctx.height)
      if (!tensor) continue

      const feeds: Record<string, import('onnxruntime-web').Tensor> = {
        [this.inputName]: new this.ort.Tensor('float32', tensor, [1, 3, PALM_SIZE, PALM_SIZE]),
      }
      let feature: Float32Array
      try {
        const outputs = await session.run(feeds)
        const o = outputs[this.outputName]
        feature = l2Normalize(this.backbone.extract(o.data as Float32Array, o.dims))
      } catch (err) {
        console.error('[palm-recognize] 推理失败', err)
        continue
      }

      const top = matchEmbedding(feature, this.entries, this.topK)
      const best = top[0] ?? null
      out.push({
        rect: handBox(landmarks),
        quad,
        score: best ? best.score : null,
        hit: best && best.score >= this.threshold ? best : null,
        embedding: feature,
      })
    }
    this.results = out
  }

  draw(o: Overlay): void {
    for (const item of this.results) {
      if (this.showRoi) {
        o.polygon(item.quad, { color: 'rgba(255,209,102,0.9)', width: 1.5, dash: [5, 3] })
      }

      if (item.hit) {
        o.rect(item.rect, { color: '#35c17b', width: 2 })
        o.label(item.rect.x, item.rect.y, `${item.hit.name} ${(item.hit.score * 100).toFixed(1)}%`, {
          background: '#35c17b',
          foreground: '#08130c',
          font: '600 12px system-ui, sans-serif',
        })
      } else if (this.showUnknown) {
        o.rect(item.rect, { color: '#8d95a5', width: 1.5, dash: [4, 3] })
        const text = item.score === null ? '未知' : `未知 ${(item.score * 100).toFixed(1)}%`
        o.label(item.rect.x, item.rect.y, text, {
          background: 'rgba(20,24,32,0.85)',
          foreground: '#c8cfdb',
        })
      }
    }
  }

  dispose(): void {
    void this.session?.release()
    this.session = null
    this.ort = null
    this.results = []
    this.entries = []
  }
}
