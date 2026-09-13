import type { FrameSource } from './FrameSource'

/**
 * 摄像头帧源。
 *
 * 注意 getMediaStream 的设备枚举顺序：浏览器在授权前不会给出设备 label，
 * 所以首次启动后要再枚举一次才能拿到可读的设备名。
 */
export class CameraSource implements FrameSource {
  readonly kind = 'camera' as const
  readonly live = true
  readonly element: HTMLVideoElement

  private stream: MediaStream | null = null
  private currentDeviceId = ''

  constructor(video: HTMLVideoElement) {
    this.element = video
  }

  get width(): number {
    return this.element.videoWidth
  }

  get height(): number {
    return this.element.videoHeight
  }

  get ready(): boolean {
    return this.stream !== null && this.element.readyState >= 2 && this.width > 0
  }

  get deviceId(): string {
    return this.currentDeviceId
  }

  async start(deviceId?: string): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('当前浏览器不支持摄像头（getUserMedia 不可用），请改用图片模式')
    }

    this.stop()

    const constraints: MediaStreamConstraints = {
      audio: false,
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    }

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints)
    } catch (err) {
      throw new Error(describeCameraError(err))
    }

    this.stream = stream
    this.currentDeviceId = deviceId ?? stream.getVideoTracks()[0]?.getSettings().deviceId ?? ''
    this.element.srcObject = stream

    // 必须等 metadata 才能拿到 videoWidth/Height，否则叠加层会按 0 算映射
    if (this.element.readyState < 1) {
      await new Promise<void>((resolve) => {
        this.element.addEventListener('loadedmetadata', () => resolve(), { once: true })
      })
    }
    await this.element.play()
  }

  stop(): void {
    for (const track of this.stream?.getTracks() ?? []) track.stop()
    this.stream = null
    this.element.srcObject = null
  }

  /** 列出可用摄像头。授权前 label 为空，此时用「摄像头 N」兜底。 */
  static async listDevices(): Promise<{ deviceId: string; label: string }[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return []
    const devices = await navigator.mediaDevices.enumerateDevices()
    return devices
      .filter((d) => d.kind === 'videoinput')
      .map((d, i) => ({
        deviceId: d.deviceId,
        label: d.label || `摄像头 ${i + 1}`,
      }))
  }
}

function describeCameraError(err: unknown): string {
  const name = err instanceof DOMException ? err.name : ''
  switch (name) {
    case 'NotAllowedError':
      return '摄像头权限被拒绝。请在浏览器地址栏的权限设置里允许摄像头后重试。'
    case 'NotFoundError':
    case 'OverconstrainedError':
      return '找不到可用的摄像头设备。'
    case 'NotReadableError':
      return '摄像头被其他程序占用（如其他浏览器标签页或会议软件）。'
    case 'SecurityError':
      return '当前页面不是安全上下文，浏览器禁止访问摄像头。请用 https 或 localhost 打开。'
    default:
      return err instanceof Error ? err.message : String(err)
  }
}
