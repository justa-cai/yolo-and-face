import type { Overlay } from '../render/Overlay'
import type { FrameSource } from '../source/FrameSource'
import type { FrameCtx, VisionTask } from '../tasks/types'
import { Ema, Throttle } from '../util/throttler'

/**
 * 默认的推理输入长边上限。
 *
 * 这些模型的内部输入其实都固定在 192~320 上下，喂 1280×720 只是让「把整帧搬进 WASM」
 * 这一步白白多搬 7 倍的像素。640 是实测下来精度和耗时比较平衡的一档：
 * 人脸对齐要的 112×112 裁剪、姿态要的手腕脚踝，在这个尺度上都还够清楚。
 */
const DEFAULT_INPUT_LONG_EDGE = 640

export interface TaskTiming {
  id: string
  name: string
  /** 单次推理耗时（滑动平均），毫秒 */
  inferMs: number
  /** 实际推理频率，Hz */
  inferHz: number
  /** 当前是否正在跑（异步任务上一帧还没回来） */
  busy: boolean
}

export interface FrameTiming {
  /** 渲染帧率 */
  fps: number
  /** 整轮绘制耗时，毫秒 */
  drawMs: number
  tasks: TaskTiming[]
}

/**
 * 推理循环。
 *
 * 一帧的结构是「先 infer 再统一 draw」：infer 可以按任务各自降频（甚至异步未回就跳过），
 * 而 draw 每帧都执行，用上一次的结果，「画面跟手」和「算得慢」因此互不拖累。
 *
 * 两个把「全开就掉到 3fps」救回来的机制（实测见 M6）：
 *
 * 1. **输入降采样**：1280×720 直接喂 MediaPipe，光是把整帧搬进 WASM 就够呛。
 *    调度器每帧把画面等比压到任务要求的长边上限（默认 640），再交给 `ctx.element`。
 *    关键点是这个缩放**对语义无影响**：关键点、检测框都是归一化坐标，画的时候乘回源尺寸，
 *    所以降采样只影响模型看到的细节，不影响叠加层对齐。
 *
 * 2. **每帧的推理预算**：以前是「所有到点的任务在同一帧里排队跑完」，于是
 *    单个任务的耗时直接相加——人脸 20ms + 姿态 46ms + 检测 122ms + 分类 54ms ≈ 240ms/帧。
 *    现在每帧最多花 `inferenceBudgetMs` 去**开新任务**（异步任务按启动时刻算），
 *    并且用一个轮转游标保证没人被饿死。贵的任务自然会各自占一帧，便宜的任务一帧能跑好几个。
 *
 * 配套改动：`shared` 不再每帧清空。跨帧保留最后一份结果，人脸识别因此可以和关键点
 * 落在不同帧上——静态图更是必须如此（关键点只算一次，识别可能排在下一帧）。
 * 换源时由 `resetForNewSource()` 显式作废。
 */
export class Scheduler {
  private tasks: readonly VisionTask[] = []
  private source: FrameSource | null = null
  private running = false
  private rafId = 0
  private startedAt = performance.now()

  /** 跨任务共享的中间结果；换源时清空，其余时候保留最后一帧的值 */
  private readonly shared = new Map<string, unknown>()

  private readonly throttles = new Map<string, Throttle>()
  private readonly inferMs = new Map<string, Ema>()
  private readonly inferHz = new Map<string, Ema>()
  private readonly lastInferAt = new Map<string, number>()
  private readonly busy = new Set<string>()
  /** 静态图已经推理过的任务，避免对着同一张图反复算 */
  private readonly stillDone = new Set<string>()

  /** 每帧最多用来「开新任务」的毫秒数。取 30fps 里的一半，留够绘制和浏览器自己的开销。 */
  private readonly inferenceBudgetMs = 16

  /** 轮转游标：下一帧从哪个任务开始试，保证预算不够时后面的任务也能轮到 */
  private rrCursor = 0

  /** 降采样用的离屏画布，按需创建、跨帧复用 */
  private scratch: HTMLCanvasElement | null = null

  private readonly frameMs = new Ema(0.08)
  private frameFps = 0
  private lastFrameAt = 0
  private lastReport = 0

  constructor(
    private readonly overlay: Overlay,
    private readonly onTiming: (t: FrameTiming) => void,
  ) {}

  setTasks(tasks: readonly VisionTask[]): void {
    this.tasks = tasks
    const ids = new Set(tasks.map((t) => t.id))
    for (const id of this.throttles.keys()) {
      if (!ids.has(id)) this.throttles.delete(id)
    }
  }

  setSource(source: FrameSource | null): void {
    this.source = source
    this.resetForNewSource()
  }

  /** 换图/换设备/新任务启用后调用，让静态图重算一次 */
  resetForNewSource(): void {
    this.stillDone.clear()
    this.shared.clear()
    for (const t of this.throttles.values()) t.reset()
  }

  start(): void {
    if (this.running) return
    this.running = true
    this.lastFrameAt = 0
    this.rafId = requestAnimationFrame(this.tick)
  }

  stop(): void {
    this.running = false
    if (this.rafId) cancelAnimationFrame(this.rafId)
    this.rafId = 0
  }

  private tick = (now: number): void => {
    if (!this.running) return
    this.rafId = requestAnimationFrame(this.tick)

    const source = this.source
    if (!source || !source.ready) return

    this.overlay.setSourceSize(source.width, source.height)
    if (!this.overlay.beginFrame()) return

    // 渲染帧率：只看两次 rAF 的间隔
    if (this.lastFrameAt > 0) {
      const dt = now - this.lastFrameAt
      if (dt > 0 && dt < 1000) this.frameMs.push(dt)
      this.frameFps = this.frameMs.current > 0 ? 1000 / this.frameMs.current : 0
    }
    this.lastFrameAt = now

    const still = !source.live
    // 降采样是「访问时才做」：本帧一个任务都不用跑（都被节流挡住或已经跑过了）时，
    // 就不该白白把 1280×720 重画一遍。实测这步不省的话，单开人脸关键点这种 8ms 的任务
    // 也能把渲染帧率从 60 拖到 16。
    let input: { element: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement; width: number; height: number } | null = null
    const ensureInput = () => {
      input ??= this.prepareInput(source, still)
      return input
    }
    const ctx: FrameCtx = {
      source,
      get element() {
        return ensureInput().element
      },
      get elementW() {
        return ensureInput().width
      },
      get elementH() {
        return ensureInput().height
      },
      width: source.width,
      height: source.height,
      timestampMs: now - this.startedAt,
      still,
      shared: this.shared,
    }

    this.runInference(ctx, still, now)

    const t0 = performance.now()
    for (const task of this.tasks) task.draw(this.overlay)
    const drawMs = performance.now() - t0

    if (now - this.lastReport > 220) {
      this.lastReport = now
      this.onTiming({
        fps: this.frameFps,
        drawMs,
        tasks: this.tasks.map((t) => ({
          id: t.id,
          name: t.name,
          inferMs: this.inferMs.get(t.id)?.current ?? 0,
          inferHz: this.inferHz.get(t.id)?.current ?? 0,
          busy: this.busy.has(t.id),
        })),
      })
    }
  }

  /**
   * 按任务要求的最大长边把画面压一次。压出来的画布跨帧复用，尺寸变了才重建。
   * 源画面本来就够小、或者所有任务都算完了，就直接把原始元素交出去，不做任何拷贝。
   */
  private prepareInput(
    source: FrameSource,
    still: boolean,
  ): { element: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement; width: number; height: number } {
    const original = { element: source.element, width: source.width, height: source.height }
    const longEdge = Math.max(source.width, source.height)
    if (longEdge === 0) return original

    let limit = Infinity
    for (const task of this.tasks) {
      if (still && this.stillDone.has(task.id)) continue
      const cap = task.maxInputLongEdge ?? DEFAULT_INPUT_LONG_EDGE
      if (cap > 0 && cap < limit) limit = cap
    }
    if (!Number.isFinite(limit) || longEdge <= limit) return original

    const scale = limit / longEdge
    const w = Math.max(1, Math.round(source.width * scale))
    const h = Math.max(1, Math.round(source.height * scale))

    const canvas = this.scratch ?? (this.scratch = document.createElement('canvas'))
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
    }
    // willReadFrequently：接下来要把像素读回 WASM，走 CPU 后备存储比 GPU 回读快得多
    const g = canvas.getContext('2d', { willReadFrequently: true })
    if (!g) return original
    try {
      g.drawImage(source.element, 0, 0, w, h)
    } catch {
      // 视频还没出帧时 drawImage 会抛 InvalidStateError
      return original
    }
    return { element: canvas, width: w, height: h }
  }

  private runInference(ctx: FrameCtx, still: boolean, now: number): void {
    const n = this.tasks.length
    if (n === 0) return

    if (still) {
      // 静态图只算一次，这里**必须按注册顺序**把所有没算过的任务跑完：
      // 依赖是靠 registry 顺序 + 帧内 shared 建立的（人脸关键点在前、人脸识别在后）。
      // 要是让轮转或预算把识别排到关键点前面，它就会拿着空的 shared 跑完、
      // 然后被标记成「已算过」，这张图从此再也不会出识别结果。
      for (const task of this.tasks) {
        if (this.stillDone.has(task.id) || this.busy.has(task.id)) continue
        this.runOne(task, ctx, true)
      }
      return
    }

    const frameStart = performance.now()
    let served = 0
    // 游标必须先在循环外取值：循环里改它会让 i 的推进量翻倍，同一个任务一帧内被喂两次，
    // 而 MediaPipe 对同一路流的 timestamp 要求严格递增——重复喂同一时间戳直接报错。
    const base = this.rrCursor
    let lastServed = -1

    // 从上次停下的地方接着试，预算不够时后面的任务下一帧就能轮到
    for (let k = 0; k < n; k++) {
      const i = (base + k) % n
      const task = this.tasks[i]

      // 上一帧的异步推理还没回来，跳过本轮，避免请求堆叠把内存吃满
      if (this.busy.has(task.id)) continue

      // 本帧预算已经花完就收手，剩下的留给下一帧。但至少放行一个，
      // 否则单任务耗时超过预算时这一帧会空转，永远推进不下去。
      if (served > 0 && performance.now() - frameStart >= this.inferenceBudgetMs) break

      let throttle = this.throttles.get(task.id)
      if (!throttle) {
        throttle = new Throttle(task.minIntervalMs)
        this.throttles.set(task.id, throttle)
      }
      // due() 会消耗掉这次闸门，所以必须放在预算判断之后
      if (!throttle.due(now)) continue

      served++
      lastServed = i
      this.runOne(task, ctx, false)
    }

    if (lastServed >= 0) this.rrCursor = (lastServed + 1) % n
  }

  /** 跑一个任务并记账（耗时 / 频率 / 静态图完成标记）。调用方负责判定该不该跑。 */
  private runOne(task: VisionTask, ctx: FrameCtx, still: boolean): void {
    const t0 = performance.now()
    const finish = (): void => {
      const dt = performance.now() - t0
      const ema = this.inferMs.get(task.id) ?? new Ema(0.15)
      ema.push(dt)
      this.inferMs.set(task.id, ema)

      const prev = this.lastInferAt.get(task.id) ?? 0
      if (prev > 0) {
        const gap = performance.now() - prev
        if (gap > 0) {
          const hz = this.inferHz.get(task.id) ?? new Ema(0.15)
          hz.push(1000 / gap)
          this.inferHz.set(task.id, hz)
        }
      }
      this.lastInferAt.set(task.id, performance.now())
      if (still) this.stillDone.add(task.id)
    }

    try {
      const out = task.infer(ctx)
      if (out instanceof Promise) {
        this.busy.add(task.id)
        out
          .then(finish, (err) => {
            console.error(`[${task.id}] 推理失败`, err)
            if (still) this.stillDone.add(task.id)
          })
          .finally(() => this.busy.delete(task.id))
      } else {
        finish()
      }
    } catch (err) {
      console.error(`[${task.id}] 推理失败`, err)
      if (still) this.stillDone.add(task.id)
    }
  }
}
