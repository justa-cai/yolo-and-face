import type { FrameSource } from './FrameSource'

/** 静态图片帧源。live = false，上层只需推理一次。 */
export class ImageSource implements FrameSource {
  readonly kind = 'image' as const
  readonly live = false
  readonly element: HTMLImageElement

  /** 当前的 object URL，换图时要回收，否则会持续泄漏内存 */
  private objectUrl: string | null = null

  constructor(img: HTMLImageElement) {
    this.element = img
    img.decoding = 'async'
  }

  get width(): number {
    return this.element.naturalWidth
  }

  get height(): number {
    return this.element.naturalHeight
  }

  get ready(): boolean {
    return this.element.complete && this.width > 0
  }

  get fileName(): string {
    return this.element.alt
  }

  /** 从 File / Blob / 拖拽项加载一张图片。 */
  async load(file: Blob, name = ''): Promise<void> {
    const url = URL.createObjectURL(file)
    try {
      await new Promise<void>((resolve, reject) => {
        this.element.onload = () => resolve()
        this.element.onerror = () => reject(new Error('图片解码失败，可能不是有效的图片文件'))
        this.element.src = url
      })
    } catch (err) {
      URL.revokeObjectURL(url)
      throw err
    }
    // 新图加载成功后再回收旧 URL，避免旧图闪白
    this.releaseUrl()
    this.objectUrl = url
    this.element.alt = name
  }

  start(): Promise<void> {
    // 图片源没有"启动"的概念，加载完即可用
    return Promise.resolve()
  }

  stop(): void {
    this.releaseUrl()
    this.element.removeAttribute('src')
    this.element.alt = ''
  }

  private releaseUrl(): void {
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl)
      this.objectUrl = null
    }
  }
}
