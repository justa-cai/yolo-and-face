/** 统一的 DOM 取用口，取不到就立刻抛错，避免后面到处写 `!`。 */

function need<T extends Element>(selector: string): T {
  const el = document.querySelector<T>(selector)
  if (!el) throw new Error(`页面缺少必要的元素：${selector}`)
  return el
}

export const dom = {
  viewport: need<HTMLDivElement>('#viewport'),
  video: need<HTMLVideoElement>('#video'),
  still: need<HTMLImageElement>('#still'),
  overlay: need<HTMLCanvasElement>('#overlay'),
  placeholder: need<HTMLDivElement>('#placeholder'),

  btnCamera: need<HTMLButtonElement>('#btn-camera'),
  btnImage: need<HTMLButtonElement>('#btn-image'),
  btnStart: need<HTMLButtonElement>('#btn-start'),
  cameraField: need<HTMLLabelElement>('#camera-field'),
  cameraSelect: need<HTMLSelectElement>('#camera-select'),
  fileInput: need<HTMLInputElement>('#file-input'),

  taskList: need<HTMLDivElement>('#task-list'),
  stats: need<HTMLDivElement>('#stats'),

  faceName: need<HTMLInputElement>('#face-name'),
  btnFaceAdd: need<HTMLButtonElement>('#btn-face-add'),
  btnFaceClear: need<HTMLButtonElement>('#btn-face-clear'),
  faceNote: need<HTMLDivElement>('#face-note'),
  faceList: need<HTMLDivElement>('#face-list'),

  palmName: need<HTMLInputElement>('#palm-name'),
  btnPalmAdd: need<HTMLButtonElement>('#btn-palm-add'),
  btnPalmClear: need<HTMLButtonElement>('#btn-palm-clear'),
  palmNote: need<HTMLDivElement>('#palm-note'),
  palmList: need<HTMLDivElement>('#palm-list'),

  backendChip: need<HTMLSpanElement>('#backend-chip'),
  sourceChip: need<HTMLSpanElement>('#source-chip'),
}

export function setChip(el: HTMLElement, text: string, tone: '' | 'ok' | 'warn' | 'err' = ''): void {
  el.textContent = text
  el.classList.remove('ok', 'warn', 'err')
  if (tone) el.classList.add(tone)
}
