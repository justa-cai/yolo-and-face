import type { TaskOptionSpec, TaskOptionValue, VisionTask } from '../tasks/types'

export type StateTone = '' | 'ok' | 'warn' | 'err'

export interface TaskControlCallbacks {
  onToggle(task: VisionTask, enabled: boolean): void
  onOption(task: VisionTask, key: string, value: TaskOptionValue): void
}

export interface TaskCard {
  setState(text: string, tone?: StateTone): void
  setEnabled(enabled: boolean): void
}

/** 根据任务的 `options` 描述自动生成侧栏卡片，接入新算法时不用再动 UI 代码。 */
export class TaskControls {
  private readonly cards = new Map<string, TaskCard>()

  constructor(
    container: HTMLElement,
    tasks: readonly VisionTask[],
    private readonly cb: TaskControlCallbacks,
  ) {
    container.replaceChildren(...tasks.map((t) => this.buildCard(t)))
  }

  card(id: string): TaskCard | undefined {
    return this.cards.get(id)
  }

  private buildCard(task: VisionTask): HTMLElement {
    const root = document.createElement('div')
    root.className = 'task'
    root.dataset.id = task.id

    const head = document.createElement('label')
    head.className = 'task-head'
    head.title = task.hint

    const checkbox = document.createElement('input')
    checkbox.type = 'checkbox'
    checkbox.addEventListener('change', () => {
      root.classList.toggle('enabled', checkbox.checked)
      this.cb.onToggle(task, checkbox.checked)
    })

    const name = document.createElement('span')
    name.className = 'task-name'
    name.textContent = task.name

    const state = document.createElement('span')
    state.className = 'task-state'
    state.textContent = '未启用'

    head.append(checkbox, name, state)

    const body = document.createElement('div')
    body.className = 'task-body'
    if (task.options.length === 0) {
      const empty = document.createElement('div')
      empty.className = 'empty'
      empty.textContent = '无可调参数'
      body.append(empty)
    } else {
      for (const opt of task.options) body.append(this.buildOption(task, opt))
    }

    root.append(head, body)

    this.cards.set(task.id, {
      setState: (text, tone = '') => {
        state.textContent = text
        state.classList.remove('ok', 'warn', 'err')
        if (tone) state.classList.add(tone)
      },
      setEnabled: (enabled) => {
        checkbox.checked = enabled
        root.classList.toggle('enabled', enabled)
      },
    })

    return root
  }

  private buildOption(task: VisionTask, opt: TaskOptionSpec): HTMLElement {
    const row = document.createElement('div')
    row.className = 'task-opt'

    const label = document.createElement('label')
    label.textContent = opt.label
    row.append(label)

    if (opt.type === 'range') {
      const input = document.createElement('input')
      input.type = 'range'
      input.min = String(opt.min ?? 0)
      input.max = String(opt.max ?? 1)
      input.step = String(opt.step ?? 0.01)
      input.value = String(opt.default)

      const val = document.createElement('span')
      val.className = 'val'
      val.textContent = opt.format ? opt.format(Number(opt.default)) : String(opt.default)

      input.addEventListener('input', () => {
        const v = Number(input.value)
        val.textContent = opt.format ? opt.format(v) : String(v)
        this.cb.onOption(task, opt.key, v)
      })

      row.append(input, val)
    } else {
      const select = document.createElement('select')
      for (const c of opt.choices ?? []) {
        const o = document.createElement('option')
        o.value = c.value
        o.textContent = c.label
        select.append(o)
      }
      select.value = String(opt.default)
      select.addEventListener('change', () => this.cb.onOption(task, opt.key, select.value))
      row.append(select)
    }

    return row
  }
}
