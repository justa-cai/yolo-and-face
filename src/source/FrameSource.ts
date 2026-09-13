/** 帧源：统一摄像头与静态图片，让上层推理循环不用关心画面从哪来。 */

export type SourceKind = 'camera' | 'image'

export interface FrameSource {
  readonly kind: SourceKind
  /** 可直接交给 MediaPipe / drawImage 的媒体元素 */
  readonly element: HTMLVideoElement | HTMLImageElement
  /** 源画面像素尺寸；未就绪时为 0 */
  readonly width: number
  readonly height: number
  /** 是否已有可推理的一帧 */
  readonly ready: boolean
  /** 画面是否逐帧变化。静态图片为 false，上层只需推理一次。 */
  readonly live: boolean
  start(): Promise<void>
  stop(): void
}

/** 画面尺寸或就绪状态发生变化时通知上层（换设备、换图片后要重算映射）。 */
export type SourceChangeHandler = (source: FrameSource) => void
