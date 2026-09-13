/** 性能面板：渲染帧率、各任务的推理耗时与推理帧率。 */

export interface StatRow {
  label: string
  value: string
  /** 分组标题行，会被跨两列显示 */
  section?: boolean
}

export class StatsPanel {
  private readonly el: HTMLElement
  private rows: StatRow[] = []
  private lastPaint = 0

  constructor(el: HTMLElement) {
    this.el = el
    this.setEmpty('尚未启动')
  }

  setEmpty(text: string): void {
    const div = document.createElement('div')
    div.className = 'empty'
    div.textContent = text
    this.el.replaceChildren(div)
  }

  setRows(rows: StatRow[]): void {
    this.rows = rows
  }

  /** 每帧调用；内部限流到约 4Hz，避免面板频繁重排反过来拖慢推理。 */
  paint(now: number, force = false): void {
    if (this.rows.length === 0) return
    if (!force && now - this.lastPaint < 250) return
    this.lastPaint = now
    this.el.replaceChildren(...this.rows.flatMap((r) => this.renderRow(r)))
  }

  private renderRow(row: StatRow): HTMLElement[] {
    if (row.section) {
      const div = document.createElement('div')
      div.className = 'section'
      div.textContent = row.label
      return [div]
    }
    const dt = document.createElement('dt')
    dt.textContent = row.label
    const dd = document.createElement('dd')
    dd.textContent = row.value
    return [dt, dd]
  }
}
