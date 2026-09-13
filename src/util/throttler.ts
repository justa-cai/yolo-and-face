/** 指数滑动平均：性能面板里用它把跳动的单帧耗时抹平到可读。 */
export class Ema {
  private value = 0
  private primed = false

  constructor(private readonly alpha = 0.15) {}

  push(sample: number): number {
    if (!this.primed) {
      this.value = sample
      this.primed = true
    } else {
      this.value += this.alpha * (sample - this.value)
    }
    return this.value
  }

  get current(): number {
    return this.value
  }

  reset(): void {
    this.primed = false
    this.value = 0
  }
}

/** 按时间间隔限流的闸门。intervalMs = 0 表示每帧都放行。 */
export class Throttle {
  private last = -Infinity

  constructor(private intervalMs = 0) {}

  setInterval(ms: number): void {
    this.intervalMs = ms
  }

  /** 到点则放行并刷新计时 */
  due(now: number): boolean {
    if (now - this.last < this.intervalMs) return false
    this.last = now
    return true
  }

  reset(): void {
    this.last = -Infinity
  }
}

/** 环形计数器，用来算"每 N 帧"这类节流 */
export class FrameCounter {
  private count = 0

  constructor(private every = 1) {}

  setEvery(n: number): void {
    this.every = Math.max(1, Math.round(n))
  }

  tick(): boolean {
    this.count = (this.count + 1) % this.every
    return this.count === 0
  }

  reset(): void {
    this.count = 0
  }
}
