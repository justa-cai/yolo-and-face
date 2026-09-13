import type { EmbeddingEntry } from '../biometric/match'

export type NoteTone = '' | 'ok' | 'warn' | 'err'

export interface GalleryPanelText {
  /** 列表为空时显示的占位文字 */
  empty: string
  /** 对应算法没开启时，替代注册按钮的提示 */
  disabled: string
  /** 清空按钮的文字 */
  clear: string
}

export interface GalleryPanelCallbacks {
  onAdd(name: string): void
  onRemove(id: string): void
  onClear(): void
}

export interface GalleryPanelElements {
  nameInput: HTMLInputElement
  addButton: HTMLButtonElement
  clearButton: HTMLButtonElement
  note: HTMLDivElement
  list: HTMLDivElement
}

/**
 * 特征库侧栏，人脸库和掌纹库共用一份实现（行为完全一致，只是文案不同）。
 * 只负责显示与转发事件，真正的读写由 main 交给 EmbeddingGallery 做。
 *
 * 每条只画名字 + 注册时间 + 删除按钮：特征向量既不显示也不导出，
 * 避免给人「这是可用的身份库」的错觉。
 */
export class GalleryPanel {
  private entries: EmbeddingEntry[] = []
  private enabled = false

  constructor(
    private readonly el: GalleryPanelElements,
    private readonly cb: GalleryPanelCallbacks,
    private readonly text: GalleryPanelText,
  ) {
    el.addButton.addEventListener('click', () => {
      this.cb.onAdd(el.nameInput.value)
    })
    el.nameInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.cb.onAdd(el.nameInput.value)
    })
    el.clearButton.addEventListener('click', () => this.cb.onClear())
    this.render()
  }

  /** 对应的识别任务是否已就绪；没就绪时禁用注册（那时候拿不到特征） */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled
    this.el.addButton.disabled = !enabled
    if (!enabled) this.setNote(this.text.disabled)
    this.render()
  }

  setEntries(entries: readonly EmbeddingEntry[]): void {
    this.entries = [...entries]
    this.render()
  }

  setNote(text: string, tone: NoteTone = ''): void {
    this.el.note.textContent = text
    this.el.note.classList.remove('ok', 'warn', 'err')
    if (tone) this.el.note.classList.add(tone)
  }

  /** 注册成功后清空输入框，方便连着录第二张 */
  clearNameInput(): void {
    this.el.nameInput.value = ''
  }

  private render(): void {
    const { list, clearButton } = this.el
    clearButton.hidden = this.entries.length === 0
    clearButton.textContent = this.text.clear

    if (this.entries.length === 0) {
      list.replaceChildren(el('div', 'empty', this.enabled ? this.text.empty : '—'))
      return
    }

    const rows = this.entries.map((entry) => {
      const row = document.createElement('div')
      row.className = 'face-item'

      const meta = document.createElement('div')
      meta.className = 'face-meta'
      const name = el('span', 'face-item-name', entry.name)
      const time = el('span', 'face-item-time', formatTime(entry.createdAt))
      meta.append(name, time)

      const del = document.createElement('button')
      del.type = 'button'
      del.className = 'ghost tiny'
      del.textContent = '删除'
      del.addEventListener('click', () => this.cb.onRemove(entry.id))

      row.append(meta, del)
      return row
    })
    list.replaceChildren(...rows)
  }
}

function el(tag: string, className: string, text: string): HTMLElement {
  const node = document.createElement(tag)
  node.className = className
  node.textContent = text
  return node
}

function formatTime(ms: number): string {
  const d = new Date(ms)
  const pad = (n: number): string => String(n).padStart(2, '0')
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
