import type { FaceEntry } from './match'

const DB_NAME = 'cv-face-gallery'
const DB_VERSION = 1
const STORE = 'faces'

/**
 * 人脸库，存在 IndexedDB 里。
 *
 * 只存**特征向量**（128 个 float），不存原始人脸图片：原始图属于生物特征数据，
 * 留在浏览器里风险远大于收益，而且本项目只是功能演示，不涉及任何身份核验业务。
 * 数据完全留在本机，清空站点数据即彻底删除。
 */
export class FaceGallery {
  private db: IDBDatabase | null = null
  /** IndexedDB 不可用时（隐私模式、被策略禁用）退化到内存，至少让功能能演示 */
  private memory: FaceEntry[] | null = null

  async open(): Promise<void> {
    if (this.db || this.memory) return
    if (typeof indexedDB === 'undefined') {
      console.warn('[gallery] 当前环境没有 IndexedDB，人脸库退化为内存存储（刷新即丢失）')
      this.memory = []
      return
    }

    this.db = await new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION)
      req.onupgradeneeded = () => {
        const db = req.result
        if (!db.objectStoreNames.contains(STORE)) {
          db.createObjectStore(STORE, { keyPath: 'id' })
        }
      }
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error ?? new Error('无法打开人脸库'))
      // 另一个标签页占着旧版本连接时会一直阻塞，明确报错比静默卡住好
      req.onblocked = () => reject(new Error('人脸库被其它标签页占用，请关闭后重试'))
    })
  }

  async list(): Promise<FaceEntry[]> {
    await this.open()
    if (this.memory) return [...this.memory]
    const all = await this.run('readonly', (store) => store.getAll() as IDBRequest<FaceEntry[]>)
    // 注册时间升序，"最早注册的排前面"符合直觉
    return all.sort((a, b) => a.createdAt - b.createdAt)
  }

  async add(name: string, embedding: Float32Array): Promise<FaceEntry> {
    await this.open()
    const entry: FaceEntry = {
      id: cryptoId(),
      name: name.trim() || '未命名',
      // 复制一份：调用方通常复用同一个 buffer，直接存引用会被下一次推理覆盖
      embedding: new Float32Array(embedding),
      createdAt: Date.now(),
    }
    if (this.memory) {
      this.memory.push(entry)
      return entry
    }
    await this.run('readwrite', (store) => store.put(entry))
    return entry
  }

  async remove(id: string): Promise<void> {
    await this.open()
    if (this.memory) {
      this.memory = this.memory.filter((e) => e.id !== id)
      return
    }
    await this.run('readwrite', (store) => store.delete(id))
  }

  async clear(): Promise<void> {
    await this.open()
    if (this.memory) {
      this.memory = []
      return
    }
    await this.run('readwrite', (store) => store.clear())
  }

  /** 把「开事务 -> 发一次请求」包成 Promise，顺带统一错误处理 */
  private run<T>(mode: IDBTransactionMode, make: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const db = this.db
    if (!db) throw new Error('人脸库尚未打开')
    return new Promise<T>((resolve, reject) => {
      let tx: IDBTransaction
      try {
        tx = db.transaction(STORE, mode)
      } catch (err) {
        reject(err instanceof Error ? err : new Error(String(err)))
        return
      }
      tx.onabort = () => reject(tx.error ?? new Error('人脸库事务被中止'))
      tx.onerror = () => reject(tx.error ?? new Error('人脸库事务失败'))
      const req = make(tx.objectStore(STORE))
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error ?? new Error('人脸库操作失败'))
    })
  }
}

function cryptoId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return `f_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`
}
