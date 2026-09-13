import type { Overlay, Pt } from '../render/Overlay'
import type { FrameCtx, TaskOptionSpec, TaskOptionValue, VisionTask } from './types'

/** 演示用的骨架：一组固定归一化坐标，用来肉眼确认缩放后肢体比例没有走形。 */
const DEMO_POINTS: Pt[] = [
  { x: 0.5, y: 0.24 }, // 0 头
  { x: 0.5, y: 0.38 }, // 1 颈
  { x: 0.4, y: 0.44 }, // 2 左肩
  { x: 0.66, y: 0.48 }, // 3 右肩
  { x: 0.33, y: 0.62 }, // 4 左肘
  { x: 0.74, y: 0.66 }, // 5 右肘
  { x: 0.28, y: 0.78 }, // 6 左腕
  { x: 0.8, y: 0.82 }, // 7 右腕
  { x: 0.44, y: 0.72 }, // 8 左髋
  { x: 0.58, y: 0.72 }, // 9 右髋
  { x: 0.42, y: 0.9 }, // 10 左膝
  { x: 0.6, y: 0.9 }, // 11 右膝
]

const DEMO_BONES: readonly (readonly [number, number])[] = [
  [0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
  [1, 8], [1, 9], [8, 9], [8, 10], [9, 11],
]

/**
 * 坐标标定：不依赖任何模型，在画面上铺一层归一化网格 + 一个正圆 + 一个固定骨架。
 *
 * 用途是**校准叠加层本身**：网格边线必须与视频画面四边严丝合缝（说明 letterbox
 * 偏移算对了），正圆必须真的圆（说明纵横比没被拉扁），骨架必须横平竖直地跟随
 * 视口缩放。窗口拉伸、切换不同分辨率摄像头时，拿它一眼就能看出映射是否失真。
 */
export class CalibrationTask implements VisionTask {
  readonly id = 'calibration'
  readonly name = '坐标标定'
  readonly stage = 'landmark' as const
  readonly hint = '不加载模型，画归一化网格与正圆，用来检查叠加层坐标映射是否正确'
  readonly assets = []
  readonly minIntervalMs = 0
  readonly options: readonly TaskOptionSpec[] = [
    {
      key: 'density',
      label: '网格',
      type: 'range',
      min: 2,
      max: 20,
      step: 1,
      default: 10,
      format: (v) => `${v} 格`,
    },
    {
      key: 'scale',
      label: '骨架',
      type: 'range',
      min: 0.5,
      max: 1.6,
      step: 0.05,
      default: 1,
      format: (v) => `${v.toFixed(2)}×`,
    },
  ]

  private density = 10
  private scale = 1

  init(): Promise<void> {
    return Promise.resolve()
  }

  setOption(key: string, value: TaskOptionValue): void {
    if (key === 'density' && typeof value === 'number') this.density = Math.round(value)
    if (key === 'scale' && typeof value === 'number') this.scale = value
  }

  infer(_ctx: FrameCtx): void {
    /* 纯绘制任务，没有推理 */
  }

  draw(o: Overlay): void {
    const n = this.density
    const step = 1 / n

    // 画布四边的框：必须精确贴合视频内容区，差一点都不行
    o.rect({ x: 0, y: 0, w: 1, h: 1 }, { color: '#4c8dff', width: 2 })

    const grid: Pt[] = []
    const links: [number, number][] = []
    for (let i = 0; i <= n; i++) {
      const t = i * step
      grid.push({ x: t, y: 0 }, { x: t, y: 1 }, { x: 0, y: t }, { x: 1, y: t })
    }
    for (let i = 0; i < grid.length; i += 2) links.push([i, i + 1])
    o.skeleton(grid, links, { color: '#8d95a5', width: 1, alpha: 0.35 })

    // 正圆：用画布像素算半径再换算回归一化坐标，纵横比错了会立刻变成椭圆
    const cw = o.contentWidth
    const ch = o.contentHeight
    if (cw > 0 && ch > 0) {
      const rPx = Math.min(cw, ch) * 0.18
      const circle: Pt[] = []
      for (let i = 0; i <= 72; i++) {
        const a = (i / 72) * Math.PI * 2
        circle.push({ x: 0.5 + (Math.cos(a) * rPx) / cw, y: 0.5 + (Math.sin(a) * rPx) / ch })
      }
      o.polyline(circle, { color: '#35c17b', width: 2 })
      o.label(0.5, 0.5 - rPx / ch, `正圆 r=${rPx.toFixed(0)}px`, { color: '#35c17b' })
    }

    // 固定骨架，乘以 scale 绕画面中心缩放
    const pts = DEMO_POINTS.map((p) => ({
      x: 0.5 + (p.x - 0.5) * this.scale,
      y: 0.5 + (p.y - 0.5) * this.scale,
    }))
    o.skeleton(pts, DEMO_BONES, { color: '#e0a13a', width: 2, alpha: 0.9 })
    o.points(pts, { color: '#e0a13a', radius: 3, outline: '#0a0c10' })
  }

  dispose(): void {
    /* 无资源可释放 */
  }
}
