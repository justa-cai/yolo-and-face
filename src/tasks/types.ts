import type { Overlay } from '../render/Overlay'
import type { FrameSource } from '../source/FrameSource'

/** 任务在流水线里的角色，决定 UI 分组与默认绘制层级 */
export type TaskStage = 'landmark' | 'detect' | 'classify'

/** 任务依赖的一个静态资产，用于懒加载与体积提示 */
export interface AssetSpec {
  /** 相对页面基址的路径，例如 models/mediapipe/pose_landmarker_lite.task */
  url: string
  /** 展示名 */
  label: string
  /** 大致字节数，仅用于下载进度提示；不需要精确 */
  bytes?: number
}

export interface TaskOptionSpec {
  key: string
  label: string
  type: 'range' | 'select'
  /** range 用 */
  min?: number
  max?: number
  step?: number
  /** select 用 */
  choices?: readonly { value: string; label: string }[]
  default: number | string
  /** 数值选项的展示格式化，range 时用于显示滑杆右侧的数字 */
  format?: (v: number) => string
}

export type TaskOptionValue = number | string

/**
 * 一帧的上下文。任务之间通过 `shared` 交换中间结果——
 * 例如人脸关键点任务写入对齐后的 5 点，人脸识别任务读走去做仿射变换。
 *
 * `element` / `elementW` / `elementH` 是**降采样后**的推理输入，喂模型的活统一用它；
 * `source.element` / `width` / `height` 始终是原始分辨率的画面，只有需要像素级精度的
 * 任务（人脸对齐要抠 112×112 的裁剪）才用。归一化关键点在两个尺度下一致，绘制不受影响。
 */
export interface FrameCtx {
  readonly source: FrameSource
  /** 喂给模型的画面（长边已被调度器压到任务要求的上限） */
  readonly element: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement
  /** 推理输入的像素尺寸 */
  readonly elementW: number
  readonly elementH: number
  /** 源画面像素尺寸 */
  readonly width: number
  readonly height: number
  /** 自页面启动以来的毫秒数，单调递增（MediaPipe 对时间戳有这个要求） */
  readonly timestampMs: number
  /** 该帧是否为静态图（静态图只需推理一次） */
  readonly still: boolean
  readonly shared: Map<string, unknown>
}

/**
 * 一个可独立开关的视觉任务。
 *
 * 生命周期：用户勾选 -> init() -> 每帧 infer() -> 每帧 draw() -> 取消勾选时 dispose()。
 * infer 只负责算，draw 只负责画；两者分开是为了让 infer 可以降频而绘制保持满帧。
 */
export interface VisionTask {
  readonly id: string
  readonly name: string
  readonly stage: TaskStage
  /** 一行说明，鼠标悬停时显示 */
  readonly hint: string
  /** 依赖的模型资产，勾选后才下载 */
  readonly assets: readonly AssetSpec[]
  /** 暴露给 UI 的可调项 */
  readonly options: readonly TaskOptionSpec[]
  /** 建议的最小推理间隔（毫秒）。0 表示每帧都跑。 */
  readonly minIntervalMs: number
  /**
   * 喂给本任务的画面长边上限（像素）。调度器会把源画面等比压到这个尺寸再交给 `ctx.element`。
   * 不设则用调度器的默认值；设成 0 表示本任务必须吃原始分辨率。
   */
  readonly maxInputLongEdge?: number
  /**
   * 必须先开着的其它任务 id。启用本任务时 UI 会自动把这些一并勾上。
   * 典型场景：人脸识别要用「人脸关键点」的 478 点做对齐。
   */
  readonly dependsOn?: readonly string[]

  init(report: (text: string, fraction: number) => void): Promise<void>
  setOption(key: string, value: TaskOptionValue): void
  infer(ctx: FrameCtx): Promise<void> | void
  draw(overlay: Overlay): void
  dispose(): void
}
