/**
 * 叠加层：唯一负责「源坐标 -> 画布像素」映射的地方。
 *
 * 所有算法任务一律交回**归一化坐标**（x、y 均为 0..1，原点在画面左上角），
 * 由本类换算成画布像素。这样任务层完全不需要关心视频分辨率、视口尺寸、
 * object-fit 造成的 letterbox，也不需要在窗口缩放时重算任何东西。
 */

/** 归一化坐标点 */
export interface Pt {
  x: number
  y: number
}

/** 归一化矩形 */
export interface Rect {
  x: number
  y: number
  w: number
  h: number
}

export interface StrokeStyle {
  color?: string
  /** 线宽，单位是画布像素（已经乘过 DPR 补偿） */
  width?: number
  alpha?: number
  dash?: number[]
}

export interface PointStyle extends StrokeStyle {
  /** 点半径，画布像素 */
  radius?: number
  /** 描边色，给点加一圈深色轮廓以提高浅色背景下的可读性 */
  outline?: string
}

export interface LabelStyle extends StrokeStyle {
  font?: string
  /** 底板颜色；传 'none' 表示不画底板 */
  background?: string
  foreground?: string
  padding?: number
}

/** 排行榜面板的一行 */
export interface PanelRow {
  /** 条右侧的文本，例如 "94%  人" */
  text: string
  /** 条长依据，通常就是该类的置信度 */
  value: number
}

export interface PanelStyle {
  title?: string
  /** value 归一化到 0..max 求条长，默认 1（即 value 本身是 0..1 的置信度） */
  max?: number
  /** 条区宽度，画布像素 */
  barWidth?: number
  /** 行高，画布像素 */
  rowHeight?: number
  padding?: number
  font?: string
  titleFont?: string
  alpha?: number
  background?: string
  trackColor?: string
  barColor?: string
  foreground?: string
  titleColor?: string
}

const DEFAULT_COLOR = '#4c8dff'

export class Overlay {
  readonly canvas: HTMLCanvasElement
  private readonly ctx: CanvasRenderingContext2D
  private readonly viewport: HTMLElement

  /** 画布 CSS 像素尺寸 */
  private boxW = 0
  private boxH = 0
  /** 内容在画布里的偏移（源画面 letterbox 之后） */
  private offsetX = 0
  private offsetY = 0
  /** 已绘制内容的宽高，画布像素 */
  private contentW = 0
  private contentH = 0

  private srcW = 0
  private srcH = 0

  constructor(canvas: HTMLCanvasElement, viewport: HTMLElement) {
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('无法获取 2D 上下文')
    this.canvas = canvas
    this.ctx = ctx
    this.viewport = viewport
    canvas.width = 0
    canvas.height = 0
  }

  /** 告诉叠加层源画面的像素尺寸。图片源换图、摄像头换分辨率时调用。 */
  setSourceSize(width: number, height: number): void {
    if (width <= 0 || height <= 0) return
    if (this.srcW === width && this.srcH === height) return
    this.srcW = width
    this.srcH = height
    this.updateMapping()
  }

  /**
   * 每帧开头调用：同步画布尺寸、清屏、重算映射。
   * 返回 false 表示视口还没有尺寸（页面未布局），这一帧应当跳过。
   */
  beginFrame(): boolean {
    const dpr = window.devicePixelRatio || 1
    const boxW = this.viewport.clientWidth
    const boxH = this.viewport.clientHeight
    if (boxW === 0 || boxH === 0) return false

    const pxW = Math.max(1, Math.round(boxW * dpr))
    const pxH = Math.max(1, Math.round(boxH * dpr))
    if (this.canvas.width !== pxW || this.canvas.height !== pxH) {
      this.canvas.width = pxW
      this.canvas.height = pxH
    }

    this.boxW = boxW
    this.boxH = boxH
    this.updateMapping()

    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    this.ctx.clearRect(0, 0, boxW, boxH)
    return true
  }

  /** 视频/图片/画布三者同为 width:100%;height:100%;object-fit:contain，
   *  所以内容区就是按 contain 规则嵌在这个盒子里的那个矩形。 */
  private updateMapping(): void {
    if (this.boxW <= 0 || this.boxH <= 0 || this.srcW <= 0 || this.srcH <= 0) {
      this.offsetX = 0
      this.offsetY = 0
      this.contentW = 0
      this.contentH = 0
      return
    }
    const scale = Math.min(this.boxW / this.srcW, this.boxH / this.srcH)
    this.contentW = this.srcW * scale
    this.contentH = this.srcH * scale
    this.offsetX = (this.boxW - this.contentW) / 2
    this.offsetY = (this.boxH - this.contentH) / 2
  }

  /** 归一化点 -> 画布 CSS 像素 */
  mapX(x: number): number {
    return this.offsetX + x * this.contentW
  }

  mapY(y: number): number {
    return this.offsetY + y * this.contentH
  }

  /** 内容区在画布上的像素尺寸，画正圆/等长刻度时需要 */
  get contentWidth(): number {
    return this.contentW
  }

  get contentHeight(): number {
    return this.contentH
  }

  // ---------------- 绘制原语（全部接受归一化坐标） ----------------

  /** 矩形框。线宽不会因为画面缩放而变粗变细。 */
  rect(r: Rect, style: StrokeStyle = {}): void {
    const ctx = this.ctx
    this.applyStroke(ctx, style)
    const x = this.mapX(r.x)
    const y = this.mapY(r.y)
    const w = r.w * this.contentW
    const h = r.h * this.contentH
    ctx.strokeRect(x, y, w, h)
    ctx.globalAlpha = 1
  }

  /** 一组散点，例如人脸 478 关键点。 */
  points(pts: readonly Pt[], style: PointStyle = {}): void {
    if (pts.length === 0) return
    const ctx = this.ctx
    const radius = style.radius ?? 1.5
    const outline = style.outline
    const color = style.color ?? DEFAULT_COLOR
    const alpha = style.alpha ?? 1

    // 先铺一遍轮廓再铺实心，效果比逐点描边干净，也少一半路径开销
    if (outline) {
      ctx.globalAlpha = alpha
      ctx.fillStyle = outline
      ctx.beginPath()
      for (const p of pts) {
        const cx = this.mapX(p.x)
        const cy = this.mapY(p.y)
        ctx.moveTo(cx + radius + 0.6, cy)
        ctx.arc(cx, cy, radius + 0.6, 0, Math.PI * 2)
      }
      ctx.fill()
    }

    ctx.globalAlpha = alpha
    ctx.fillStyle = color
    ctx.beginPath()
    for (const p of pts) {
      const cx = this.mapX(p.x)
      const cy = this.mapY(p.y)
      ctx.moveTo(cx + radius, cy)
      ctx.arc(cx, cy, radius, 0, Math.PI * 2)
    }
    ctx.fill()
    ctx.globalAlpha = 1
  }

  /** 折线。用于画网格、轨迹等不属于骨骼语义的连线。 */
  polyline(pts: readonly Pt[], style: StrokeStyle = {}): void {
    if (pts.length < 2) return
    const ctx = this.ctx
    this.applyStroke(ctx, style)
    ctx.lineCap = 'round'
    ctx.beginPath()
    pts.forEach((p, i) => {
      const cx = this.mapX(p.x)
      const cy = this.mapY(p.y)
      if (i === 0) ctx.moveTo(cx, cy)
      else ctx.lineTo(cx, cy)
    })
    ctx.stroke()
    ctx.globalAlpha = 1
  }

  /** 骨骼连线。connections 是 [起点下标, 终点下标] 的列表。 */
  skeleton(
    pts: readonly Pt[],
    connections: readonly (readonly [number, number])[],
    style: StrokeStyle = {},
  ): void {
    const ctx = this.ctx
    this.applyStroke(ctx, style)
    ctx.lineCap = 'round'
    ctx.beginPath()
    for (const [a, b] of connections) {
      const pa = pts[a]
      const pb = pts[b]
      if (!pa || !pb) continue
      ctx.moveTo(this.mapX(pa.x), this.mapY(pa.y))
      ctx.lineTo(this.mapX(pb.x), this.mapY(pb.y))
    }
    ctx.stroke()
    ctx.globalAlpha = 1
  }

  /**
   * 文字标签，默认挂在 (x, y) 的左上角；贴到画布上/右边缘时自动翻转，
   * 保证文字始终完整可见。
   */
  label(x: number, y: number, text: string, style: LabelStyle = {}): void {
    const ctx = this.ctx
    const font = style.font ?? '12px system-ui, sans-serif'
    const padding = style.padding ?? 4
    const alpha = style.alpha ?? 1

    ctx.font = font
    ctx.textBaseline = 'middle'
    ctx.textAlign = 'left'
    const textW = ctx.measureText(text).width
    const lines = text.split('\n')
    const lineH = 14
    const boxW = textW + padding * 2
    const boxH = lineH * lines.length + padding

    let bx = this.mapX(x)
    let by = this.mapY(y)
    if (bx + boxW > this.boxW) bx = this.boxW - boxW
    if (by + boxH > this.boxH) by = by - boxH
    bx = Math.max(0, bx)
    by = Math.max(0, by)

    ctx.globalAlpha = alpha
    if (style.background !== 'none') {
      ctx.fillStyle = style.background ?? 'rgba(10,12,16,0.72)'
      ctx.fillRect(bx, by, boxW, boxH)
    }

    ctx.fillStyle = style.foreground ?? style.color ?? '#e7eaf0'
    lines.forEach((line, i) => {
      ctx.fillText(line, bx + padding, by + padding + lineH * (i + 0.5))
    })
    ctx.globalAlpha = 1
  }

  /**
   * 排行榜面板：每行「进度条 + 文本」，用于分类结果这类需要看 Top-N 分布的场景。
   *
   * 面板宽度按最长文本自适应，贴到画布右/下边缘时整体翻转，保证不越界。
   */
  panel(x: number, y: number, rows: readonly PanelRow[], style: PanelStyle = {}): void {
    if (rows.length === 0 || this.contentW <= 0) return
    const ctx = this.ctx
    const pad = style.padding ?? 8
    const rowH = style.rowHeight ?? 18
    const barW = style.barWidth ?? 72
    const gap = 8
    const font = style.font ?? '12px system-ui, sans-serif'
    const titleFont = style.titleFont ?? '600 12px system-ui, sans-serif'
    const max = style.max ?? 1

    ctx.font = font
    let textW = 0
    for (const row of rows) textW = Math.max(textW, ctx.measureText(row.text).width)
    const title = style.title
    if (title) {
      ctx.font = titleFont
      textW = Math.max(textW, ctx.measureText(title).width)
    }

    const wPx = pad * 2 + barW + gap + textW
    const hPx = pad * 2 + (title ? rowH : 0) + rows.length * rowH

    let bx = this.mapX(x)
    let by = this.mapY(y)
    if (bx + wPx > this.boxW) bx = this.boxW - wPx
    if (by + hPx > this.boxH) by = by - hPx
    bx = Math.max(0, bx)
    by = Math.max(0, by)

    ctx.globalAlpha = style.alpha ?? 1
    ctx.fillStyle = style.background ?? 'rgba(10,12,16,0.78)'
    ctx.fillRect(bx, by, wPx, hPx)

    ctx.textBaseline = 'middle'
    ctx.textAlign = 'left'
    let cy = by + pad

    if (title) {
      ctx.font = titleFont
      ctx.fillStyle = style.titleColor ?? '#e7eaf0'
      ctx.fillText(title, bx + pad, cy + rowH / 2)
      cy += rowH
    }

    ctx.font = font
    for (const row of rows) {
      const ratio = Math.max(0, Math.min(1, max > 0 ? row.value / max : 0))
      const tx = bx + pad
      const ty = cy + rowH / 2
      const bh = Math.max(4, rowH - 8)

      // 条槽：即使值为 0 也留一条暗底，行与行之间才有对齐感
      ctx.fillStyle = style.trackColor ?? 'rgba(255,255,255,0.10)'
      ctx.fillRect(tx, ty - bh / 2, barW, bh)
      ctx.fillStyle = style.barColor ?? '#4c8dff'
      ctx.fillRect(tx, ty - bh / 2, barW * ratio, bh)

      ctx.fillStyle = style.foreground ?? '#e7eaf0'
      ctx.fillText(row.text, tx + barW + gap, ty)
      cy += rowH
    }
    ctx.globalAlpha = 1
  }

  private applyStroke(ctx: CanvasRenderingContext2D, style: StrokeStyle): void {
    ctx.globalAlpha = style.alpha ?? 1
    ctx.strokeStyle = style.color ?? DEFAULT_COLOR
    ctx.lineWidth = style.width ?? 2
    ctx.setLineDash(style.dash ?? [])
  }
}
