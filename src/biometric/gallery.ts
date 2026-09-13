import type { EmbeddingEntry } from './match'

export interface GalleryInfo {
  /** IndexedDB 数据库名 */
  db: string
  /** 对象仓库名 */
  store: string
  /** 出错信息里用的中文名，例如「人脸库」 */
  label: string
}

/**
 * 人脸库。库名与仓库名是**既定值**，改名等于把用户已经注册好的人脸弄丢。
 * `cv-face-gallery` 从 M5 起就在用，别动。
 */
export const FACE_GALLERY: GalleryInfo = {
  db: 'cv-face-gallery',
  store: 'faces',
  label: '人脸库',
}

/** 掌纹库。与人物人脸库分开存：两者的特征维度和语义都不同，绝不能混在一起比对。 */
export const PALM_GALLERY: GalleryInfo = {
  db: 'cv-palm-gallery',
  store: 'palms',
  label: '掌纹库',
}

/**
 * 特征库，存在 IndexedDB 里。人脸与掌纹各用一个实例、各自一个库。
 *
 * 只存**特征向量**，不存原始图片：原始图属于生物特征数据，留在浏览器里风险远大于收益，
 * 而且本项目只是功能演示，不涉及任何身份核验业务。数据完全留在本机，
 * 清空站点数据即彻底删除。
 *
 * 两个库的名字是写死的既定值，**改名等于把用户已经注册好的数据弄丢**，别顺手改。
 */
export class EmbeddingGallery {
  private db: IDBDatabase | null = null
  /** IndexedDB 不可用时（隐私模式、被策略禁用）退化到内存，至少让功能能演示 */
  private memory: EmbeddingEntry[] | null = null

  constructor(private readonly info: GalleryInfo) {}

  async open(): Promise<void> {
    if (this.db || this.memory) return
    if (typeof indexedDB === 'undefined') {
      console.warn(`[gallery] 当前环境没有 IndexedDB，${this.info.label}退化为内存存储（刷新即丢失）`)
      this.memory = []
      return
    }

    const { db: name, store, label } = this.info
    this.db = await new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(name, 1)
      req.onupgradeneeded = () => {
        const db = req.result
        if (!db.objectStoreNames.contains(store)) {
          db.createObjectStore(store, { keyPath: 'id' })
        }
      }
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error ?? new Error(`无法打开${label}`))
      // 另一个标签页占着旧版本连接时会一直阻塞，明确报错比静默卡住好
      req.onblocked = () => reject(new Error(`${label}被其它标签页占用，请关闭后重试`))
    })
  }

  async list(): Promise<EmbeddingEntry[]> {
    await this.open()
    if (this.memory) return [...this.memory]
    const all = await this.run('readonly', (store) => store.getAll() as IDBRequest<EmbeddingEntry[]>)
    // 注册时间升序，"最早注册的排前面"符合直觉
    return all.sort((a, b) => a.createdAt - b.createdAt)
  }

  async add(name: string, embedding: Float32Array): Promise<EmbeddingEntry> {
    await this.open()
    const entry: EmbeddingEntry = {
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
    if (!db) throw new Error(`${this.info.label}尚未打开`)
    const { store: storeName, label } = this.info
    return new Promise<T>((resolve, reject) => {
      let tx: IDBTransaction
      try {
        tx = db.transaction(storeName, mode)
      } catch (err) {
        reject(err instanceof Error ? err : new Error(String(err)))
        return
      }
      tx.onabort = () => reject(tx.error ?? new Error(`${label}事务被中止`))
      tx.onerror = () => reject(tx.error ?? new Error(`${label}事务失败`))
      const req = make(tx.objectStore(storeName))
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error ?? new Error(`${label}操作失败`))
    })
  }
}

function cryptoId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return `e_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`
}
