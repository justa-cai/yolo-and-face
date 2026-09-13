import type { OrtModule } from '../runtime/ort'
import { createOrtSession, loadOrt } from '../runtime/ort'
import type { Overlay } from '../render/Overlay'
import { alignFaceToTensor, FACE_SIZE } from '../util/imageOps'
import {
  FACE_THRESHOLD,
  l2Normalize,
  matchEmbedding,
  type EmbeddingEntry,
  type EmbeddingMatch,
} from '../biometric/match'
import type { EmbeddingGallery } from '../biometric/gallery'
import {
  fivePointLandmarks,
  landmarkBox,
  SHARED_FACE_BOXES,
  SHARED_FACE_LANDMARKS,
} from './faceLandmark'
import type { AssetSpec, BiometricTask, FrameCtx, TaskOptionSpec, TaskOptionValue } from './types'
import type { Pt, Rect } from '../render/Overlay'

const MODEL: AssetSpec = {
  url: 'models/onnx/face_recognition_sface_2021dec.onnx',
  label: '人脸识别模型 (SFace)',
  bytes: 36900000,
}

/** 一帧里一张脸的识别结果 */
export interface FaceIdentification {
  rect: Rect
  /** 与人脸库里最像的那条的余弦相似度；库为空时为 null */
  score: number | null
  /** 命中的人脸库条目；未过阈值时为 null */
  hit: EmbeddingMatch | null
  /** 128 维特征，注册人脸时要用 */
  embedding: Float32Array
}

/**
 * 人脸识别 1:N（SFace + ONNX Runtime Web）。
 *
 * 复用「人脸关键点」任务的 478 点结果做五点左右对齐——这是同类方案的标准做法：
 * 直接把整帧缩到 112×112 送进去，姿态稍微一变特征就飘了，同一个人会被判成不同人。
 * 所以这个任务依赖 face-landmark 一起开启（`dependsOn`），UI 会自动带上它。
 */
export class FaceRecognizeTask implements BiometricTask {
  readonly id = 'face-recognize'
  readonly name = '人脸识别 1:N'
  readonly stage = 'classify' as const
  readonly hint = 'SFace 128 维特征 + 余弦匹配。会自动开启「人脸关键点」用于人脸对齐（依赖它的 478 点）'
  readonly dependsOn = ['face-landmark'] as const
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
      label: '判定阈值',
      type: 'range',
      min: 0.1,
      max: 0.8,
      step: 0.005,
      default: FACE_THRESHOLD,
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

  private readonly gallery: EmbeddingGallery
  private entries: EmbeddingEntry[] = []
  private results: FaceIdentification[] = []

  private threshold = FACE_THRESHOLD
  private topK = 1
  private showUnknown = true

  constructor(gallery: EmbeddingGallery) {
    this.gallery = gallery
  }

  async init(report: (text: string, fraction: number) => void): Promise<void> {
    const { session, backend } = await createOrtSession(MODEL, report)
    this.session = session
    this.backend = backend
    this.inputName = session.inputNames[0]
    this.outputName = session.outputNames[0]
    this.ort = await loadOrt()
    await this.reloadGallery()
    report('就绪', 1)
  }

  /** 注册/删除人脸后调用，让下一帧用新的人脸库 */
  async reloadGallery(): Promise<void> {
    try {
      this.entries = await this.gallery.list()
    } catch (err) {
      console.error('[face-recognize] 读取人脸库失败', err)
      this.entries = []
    }
  }

  get backendLabel(): string {
    return this.backend === 'webgpu' ? 'ONNX WebGPU' : 'ONNX WASM'
  }

  /** 人脸这边只有一个骨干（SFace），固定返回它的名字；掌纹那边是可切换的 */
  readonly backboneLabel = 'SFace'

  /** 人脸库只有一个特征空间，不会出现维度对不上的情况 */
  readonly galleryWarning = null

  /** 最近一帧里第一张脸的特征，供给「注册当前人脸」用 */
  captureEmbedding(): Float32Array | null {
    return this.results[0]?.embedding ?? null
  }

  /** 最近一帧识别出的人脸数，UI 用来提示「画面里没人脸」 */
  get faceCount(): number {
    return this.results.length
  }

  setOption(key: string, value: TaskOptionValue): void {
    if (key === 'threshold' && typeof value === 'number') this.threshold = value
    if (key === 'topk' && typeof value === 'number') this.topK = value
    if (key === 'unknown') this.showUnknown = value === 'label'
  }

  async infer(ctx: FrameCtx): Promise<void> {
    const session = this.session
    if (!session || !this.ort) return

    const faces = ctx.shared.get(SHARED_FACE_LANDMARKS) as Pt[][] | undefined
    if (!faces || faces.length === 0) {
      this.results = []
      return
    }
    const boxes = ctx.shared.get(SHARED_FACE_BOXES) as Rect[] | undefined

    const out: FaceIdentification[] = []
    for (let i = 0; i < faces.length; i++) {
      const five = fivePointLandmarks(faces[i])
      if (!five) continue

      // 这里刻意用原始分辨率的画面而不是调度器降采样后的 ctx.element：
      // 输出只有 112×112，源画面越清楚，重采样后越接近 SFace 训练时的成像条件。
      // 代价也不高——drawImage 只采样人脸那一小块，与整帧大小无关。
      const tensor = alignFaceToTensor(ctx.source.element, five, ctx.width, ctx.height)
      if (!tensor) continue

      const feeds: Record<string, import('onnxruntime-web').Tensor> = {
        [this.inputName]: new this.ort.Tensor('float32', tensor, [1, 3, FACE_SIZE, FACE_SIZE]),
      }
      let feature: Float32Array
      try {
        const outputs = await session.run(feeds)
        feature = l2Normalize(outputs[this.outputName].data as Float32Array)
      } catch (err) {
        console.error('[face-recognize] 推理失败', err)
        continue
      }

      const top = matchEmbedding(feature, this.entries, this.topK)
      const best = top[0] ?? null
      out.push({
        rect: boxes?.[i] ?? landmarkBox(faces[i]),
        score: best ? best.score : null,
        hit: best && best.score >= this.threshold ? best : null,
        embedding: feature,
      })
    }
    this.results = out
  }

  draw(o: Overlay): void {
    for (const item of this.results) {
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
