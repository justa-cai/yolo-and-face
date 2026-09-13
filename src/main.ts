import './style.css'

import { FaceGallery } from './face/FaceGallery'
import { Overlay } from './render/Overlay'
import { Scheduler, type FrameTiming } from './runtime/scheduler'
import { CameraSource } from './source/CameraSource'
import type { FrameSource } from './source/FrameSource'
import { ImageSource } from './source/ImageSource'
import { FaceRecognizeTask } from './tasks/faceEmbed'
import { createTasks } from './tasks/registry'
import type { TaskOptionValue, VisionTask } from './tasks/types'
import { TaskControls } from './ui/controls'
import { dom, setChip } from './ui/dom'
import { GalleryPanel } from './ui/gallery'
import { currentBackendLabel } from './runtime/mediapipe'
import { StatsPanel, type StatRow } from './ui/stats'

class App {
  private readonly overlay = new Overlay(dom.overlay, dom.viewport)
  private readonly camera = new CameraSource(dom.video)
  private readonly image = new ImageSource(dom.still)
  private readonly gallery = new FaceGallery()
  private readonly tasks = createTasks(this.gallery)
  private readonly enabledIds = new Set<string>()
  private readonly controls: TaskControls
  private readonly galleryPanel: GalleryPanel
  private readonly scheduler: Scheduler
  private readonly stats = new StatsPanel(dom.stats)

  private mode: 'camera' | 'image' = 'camera'
  private source: FrameSource | null = null
  private running = false
  private busyStarting = false

  constructor() {
    this.controls = new TaskControls(dom.taskList, this.tasks, {
      onToggle: (task, on) => void this.toggleTask(task, on),
      onOption: (task, key, value) => this.setOption(task, key, value),
    })

    this.galleryPanel = new GalleryPanel(
      {
        nameInput: dom.faceName,
        addButton: dom.btnFaceAdd,
        clearButton: dom.btnFaceClear,
        note: dom.faceNote,
        list: dom.faceList,
      },
      {
        onAdd: (name) => void this.registerFace(name),
        onRemove: (id) => void this.removeFace(id),
        onClear: () => void this.clearFaces(),
      },
    )

    this.scheduler = new Scheduler(this.overlay, (t) => this.onTiming(t))

    this.bindEvents()
    this.applyBackendChip()
    this.stats.setEmpty('选择帧源并启动')
  }

  /** 人脸识别任务的实例；registry 里没有就返回 null（理论上不会发生） */
  private get recognizer(): FaceRecognizeTask | null {
    const task = this.tasks.find((t) => t.id === 'face-recognize')
    return task instanceof FaceRecognizeTask ? task : null
  }

  // ---------------------------------------------------------------- 事件绑定

  private bindEvents(): void {
    dom.btnCamera.addEventListener('click', () => this.setMode('camera'))
    dom.btnImage.addEventListener('click', () => this.setMode('image'))

    dom.btnStart.addEventListener('click', () => {
      if (this.running) this.stopSource()
      else void this.startSource()
    })

    dom.cameraSelect.addEventListener('change', () => {
      if (this.running) void this.startSource()
    })

    dom.fileInput.addEventListener('change', () => {
      const file = dom.fileInput.files?.[0]
      if (file) void this.startImage(file)
      dom.fileInput.value = ''
    })

    // 拖拽图片到画面上
    dom.viewport.addEventListener('dragover', (e) => {
      e.preventDefault()
      if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy'
    })
    dom.viewport.addEventListener('drop', (e) => {
      e.preventDefault()
      const file = [...(e.dataTransfer?.files ?? [])].find((f) => f.type.startsWith('image/'))
      if (file) {
        this.setMode('image')
        void this.startImage(file)
      }
    })

    // Ctrl+V 粘贴剪贴板里的图片
    window.addEventListener('paste', (e) => {
      const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith('image/'))
      const file = item?.getAsFile()
      if (file) {
        this.setMode('image')
        void this.startImage(file)
      }
    })

    navigator.mediaDevices?.addEventListener?.('devicechange', () => void this.refreshDevices())
  }

  private setMode(mode: 'camera' | 'image'): void {
    if (this.mode === mode) return
    this.stopSource()
    this.mode = mode
    dom.btnCamera.classList.toggle('active', mode === 'camera')
    dom.btnImage.classList.toggle('active', mode === 'image')
    dom.cameraField.hidden = mode !== 'camera'
    dom.video.hidden = mode !== 'camera'
    dom.still.hidden = mode !== 'image'
    dom.placeholder.hidden = false
    dom.placeholder.textContent =
      mode === 'camera' ? '点击「启动」打开摄像头' : '点击「图片」选择一张图片，或直接拖拽/粘贴到画面上'
    setChip(dom.sourceChip, '源 未启动')
    if (mode === 'image') dom.fileInput.click()
  }

  // ---------------------------------------------------------------- 帧源

  private async startSource(): Promise<void> {
    if (this.busyStarting) return
    this.busyStarting = true
    dom.btnStart.disabled = true
    try {
      if (this.mode === 'camera') {
        await this.startCamera()
      } else if (this.image.ready) {
        this.activate(this.image, `图片 ${this.image.fileName || ''}`.trim())
      } else {
        dom.fileInput.click()
      }
    } finally {
      this.busyStarting = false
      dom.btnStart.disabled = false
    }
  }

  private async startCamera(): Promise<void> {
    setChip(dom.sourceChip, '源 启动中…', 'warn')
    try {
      await this.camera.start(dom.cameraSelect.value || undefined)
    } catch (err) {
      setChip(dom.sourceChip, '源 启动失败', 'err')
      this.overlay.beginFrame()
      dom.placeholder.hidden = false
      dom.placeholder.textContent = err instanceof Error ? err.message : String(err)
      return
    }
    await this.refreshDevices()
    this.activate(this.camera, `摄像头 ${this.camera.width}×${this.camera.height}`)
    dom.btnStart.textContent = '停止'
  }

  private async startImage(file: File): Promise<void> {
    setChip(dom.sourceChip, '源 加载图片…', 'warn')
    try {
      await this.image.load(file, file.name)
    } catch (err) {
      setChip(dom.sourceChip, '源 图片无效', 'err')
      dom.placeholder.hidden = false
      dom.placeholder.textContent = err instanceof Error ? err.message : String(err)
      return
    }
    this.activate(this.image, `图片 ${file.name} · ${this.image.width}×${this.image.height}`)
    dom.btnStart.textContent = '停止'
  }

  /** 切换到某个已经就绪的帧源并开跑推理循环 */
  private activate(source: FrameSource, label: string): void {
    this.source = source
    this.running = true
    dom.placeholder.hidden = true
    setChip(dom.sourceChip, `源 ${label}`, 'ok')
    this.scheduler.setSource(source)
    this.scheduler.start()
  }

  private stopSource(): void {
    this.camera.stop()
    this.image.stop()
    this.scheduler.stop()
    this.scheduler.setSource(null)
    this.source = null
    this.running = false
    dom.btnStart.textContent = '启动'
    dom.placeholder.hidden = false
    dom.placeholder.textContent = '已停止'
    setChip(dom.sourceChip, '源 未启动')
    this.overlay.beginFrame()
    this.stats.setEmpty('已停止')
  }

  private async refreshDevices(): Promise<void> {
    const devices = await CameraSource.listDevices()
    const current = dom.cameraSelect.value
    dom.cameraSelect.replaceChildren(
      ...devices.map((d) => {
        const opt = document.createElement('option')
        opt.value = d.deviceId
        opt.textContent = d.label
        return opt
      }),
    )
    if (devices.some((d) => d.deviceId === current)) dom.cameraSelect.value = current
  }

  // ---------------------------------------------------------------- 任务

  private async toggleTask(task: VisionTask, on: boolean): Promise<void> {
    const card = this.controls.card(task.id)
    if (!on) {
      task.dispose()
      this.enabledIds.delete(task.id)
      this.syncTasks()
      card?.setState('未启用')
      this.scheduler.resetForNewSource()
      this.syncGalleryPanel()
      return
    }

    // 先把依赖的任务勾上：人脸识别要用「人脸关键点」的 478 点做对齐，
    // 缺了它拿不到五点关键点，一帧特征也算不出来
    for (const depId of task.dependsOn ?? []) {
      const dep = this.tasks.find((t) => t.id === depId)
      if (!dep || this.enabledIds.has(depId)) continue
      this.controls.card(depId)?.setEnabled(true)
      await this.toggleTask(dep, true)
    }

    card?.setState('加载中…', 'warn')
    try {
      await task.init((text, fraction) => {
        const pct = Number.isFinite(fraction) ? ` ${Math.round(fraction * 100)}%` : ''
        card?.setState(`${text}${pct}`, 'warn')
      })
    } catch (err) {
      console.error(`[${task.id}] 初始化失败`, err)
      card?.setState(err instanceof Error ? err.message : '初始化失败', 'err')
      card?.setEnabled(false)
      return
    }

    this.enabledIds.add(task.id)
    this.syncTasks()
    card?.setState('就绪', 'ok')
    this.syncGalleryPanel()
    // 静态图模式下让新任务立刻算一次
    this.scheduler.resetForNewSource()
  }

  // ---------------------------------------------------------------- 人脸库

  /** 人脸识别开关变化时，同步侧栏的可用状态与人脸库列表 */
  private syncGalleryPanel(): void {
    const on = this.enabledIds.has('face-recognize')
    this.galleryPanel.setEnabled(on)
    if (!on) return
    void this.refreshGallery()
  }

  /** 重新读一遍人脸库并刷新侧栏，返回当前条数 */
  private async refreshGallery(): Promise<number> {
    try {
      const entries = await this.gallery.list()
      this.galleryPanel.setEntries(entries)
      // 任务内部也留了一份缓存，注册/删除后必须让它重新读一次
      await this.recognizer?.reloadGallery()
      const backend = this.recognizer?.backendLabel
      this.galleryPanel.setNote(
        `${backend ? `后端 ${backend} · ` : ''}已注册 ${entries.length} 条`,
        'ok',
      )
      return entries.length
    } catch (err) {
      this.galleryPanel.setNote(err instanceof Error ? err.message : '读取人脸库失败', 'err')
      return 0
    }
  }

  private async registerFace(name: string): Promise<void> {
    const rec = this.recognizer
    if (!rec || !this.enabledIds.has(rec.id)) {
      this.galleryPanel.setNote('请先开启「人脸识别 1:N」', 'warn')
      return
    }
    if (!this.running || !this.source) {
      this.galleryPanel.setNote('先启动摄像头或加载一张图片', 'warn')
      return
    }

    const embedding = rec.captureEmbedding()
    if (!embedding) {
      this.galleryPanel.setNote('画面里还没有可用的人脸特征，正对镜头稍等一下再点', 'warn')
      return
    }

    try {
      const entry = await this.gallery.add(name, embedding)
      this.galleryPanel.clearNameInput()
      const total = await this.refreshGallery()
      this.galleryPanel.setNote(`已注册「${entry.name}」，共 ${total} 条`, 'ok')
      // 静态图要重算一次才会把名字画上去
      this.scheduler.resetForNewSource()
    } catch (err) {
      this.galleryPanel.setNote(err instanceof Error ? err.message : '注册失败', 'err')
    }
  }

  private async removeFace(id: string): Promise<void> {
    try {
      await this.gallery.remove(id)
      await this.refreshGallery()
      this.scheduler.resetForNewSource()
    } catch (err) {
      this.galleryPanel.setNote(err instanceof Error ? err.message : '删除失败', 'err')
    }
  }

  private async clearFaces(): Promise<void> {
    try {
      await this.gallery.clear()
      await this.refreshGallery()
      this.scheduler.resetForNewSource()
    } catch (err) {
      this.galleryPanel.setNote(err instanceof Error ? err.message : '清空失败', 'err')
    }
  }

  private setOption(task: VisionTask, key: string, value: TaskOptionValue): void {
    task.setOption(key, value)
    // 参数可能影响静态图的结果，重算一次
    if (this.source && !this.source.live) this.scheduler.resetForNewSource()
  }

  private syncTasks(): void {
    this.scheduler.setTasks(this.tasks.filter((t) => this.enabledIds.has(t.id)))
  }

  // ---------------------------------------------------------------- 性能面板

  private onTiming(t: FrameTiming): void {
    const rows: StatRow[] = [
      { label: '渲染帧率', value: this.running ? `${t.fps.toFixed(1)} fps` : '—' },
      { label: '绘制耗时', value: `${t.drawMs.toFixed(2)} ms` },
    ]
    if (t.tasks.length > 0) {
      rows.push({ label: '推理', value: '', section: true })
      for (const task of t.tasks) {
        const state = task.busy ? ' · 计算中' : ''
        rows.push({
          label: task.name,
          value: `${task.inferMs.toFixed(1)} ms · ${task.inferHz.toFixed(1)} Hz${state}`,
        })
      }
    }
    this.stats.setRows(rows)
    this.stats.paint(performance.now())
  }

  private applyBackendChip(): void {
    const backend = currentBackendLabel()
    setChip(dom.backendChip, `后端 ${backend}`, backend.includes('WebGL') ? 'ok' : 'warn')
  }

  /** 供控制台排查用 */
  describe(): string {
    return this.tasks.map((t) => `${t.id}(${t.stage})`).join(', ')
  }
}

const app = new App()
;(globalThis as unknown as { app: App }).app = app
