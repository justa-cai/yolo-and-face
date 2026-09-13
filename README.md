# 浏览器端实时 CV 推理与可视化

纯前端的计算机视觉演示站：打开网页，用**摄像头**或**本地图片**作为输入，实时跑
人脸 / 骨骼点 / 物体三类算法，并把结果叠加画在画面上。

**所有推理都在浏览器里完成，没有任何后端**。模型和 WASM 运行时全部自托管，
首次加载后可以完全断网运行。

> 在线演示：<https://justa-cai.github.io/yolo-and-face/>
> （GitHub Pages 配不了 COOP/COEP，线上跑的是单线程降级路径，功能完整、速度偏慢；
> 想要最好性能请按下面的「离线 / 内网部署」自托管。）

## 能做什么

| 任务 | 模型 | 说明 |
|---|---|---|
| 人脸检测 | BlazeFace 短距 | 人脸框 + 6 个关键点 |
| 人脸关键点 | MediaPipe FaceLandmarker | 478 个 3D 关键点，含虹膜与 52 项表情系数 |
| 姿态 33 点 | MediaPipe PoseLandmarker | **单人**，Lite / Full 可切换 |
| 物体检测 | EfficientDet-Lite0 | COCO 80 类，框 + 类别 + 置信度 |
| 图像分类 | EfficientNet-Lite0 | ImageNet 1000 类，整帧 Top-N 排行榜 |
| 人脸识别 1:N | SFace (ONNX) | 128 维特征 + 余弦匹配，可注册多张人脸 |
| 坐标标定 | 无模型 | 归一化网格 + 正圆，用来核对叠加层坐标映射 |

每个任务独立开关，勾选后才去下载它自己的模型（懒加载）。参数（阈值、Top-N、点径……）
都在卡片里实时可调。

## 快速开始

```bash
git clone git@github.com:justa-cai/yolo-and-face.git
cd yolo-and-face
pnpm install
pnpm fetch-assets     # 把模型与 WASM 运行时拉到 public/（约 116MB，只需一次）
pnpm dev              # http://127.0.0.1:5173
```

`fetch-assets` 是**幂等**的：已存在且大小对得上就跳过，可以反复跑。

```bash
pnpm fetch-assets --all-wasm    # 额外拷 MediaPipe 的 nosimd 回退版（+10.5MB）
pnpm fetch-assets --with-webgpu # 额外拷 ORT 的 WebGPU(jsep) 运行时，+28MB
```

默认只拷一份 `vision_wasm_internal.(js|wasm)`。`--all-wasm` 还会带上
`vision_wasm_nosimd_internal.*`（给不支持 SIMD 的老浏览器兜底）。
勾了 `--all-wasm` 还会多拷一份 `vision_wasm_module_internal.*`，**那份用不上**
（见「性能」一节的说明），介意体积的话删掉即可。

## 离线 / 内网部署

```bash
pnpm build            # 产出 dist/，约 128MB
```

`dist/` 是**完全自包含**的，整个目录拷到任何地方、任何静态服务器都能直接打开：

```bash
pnpm serve:dist       # 等价于 python3 -m http.server 8899 --directory dist
```

`base: './'` 保证所有资源都走相对路径，所以既能以根目录打开，也能挂在子路径下。

### 想要更好的性能：加上 COOP/COEP

```nginx
location / {
  add_header Cross-Origin-Opener-Policy   "same-origin";
  add_header Cross-Origin-Embedder-Policy "require-corp";
}
```

带上这两个响应头后页面就处于**跨源隔离**状态，`SharedArrayBuffer` 可用，
onnxruntime-web 会把 `numThreads` 开到 4（人脸识别的 WASM 推理因此变快）。

**这是增强项，不是必需项**：不配也能跑，只是慢一点。开发服务器（`pnpm dev`）
已经默认带上了这两个头。

> ⚠️ MediaPipe 那条路径**不**通过这个开关提速。`FilesetResolver` 的第二参数
> `useModule` 看起来是「开多线程」，实际选的是给 **Worker** 用的 ESM 打包变体，
> 在主线程上会被当普通脚本加载，直接 `SyntaxError: Cannot use 'import.meta'
> outside a module`。真要吃 MediaPipe 的多线程必须把任务整体搬进 Web Worker，
> 见下面「还没做」。

### GitHub Pages

```bash
pnpm deploy:pages            # 构建并推送到 origin/gh-pages
pnpm deploy:pages --dry-run  # 只构建
```

需要仓库已经 `git init` 并配好 remote。**Pages 配不了 COOP/COEP**，所以线上跑的是
未隔离的降级路径（单线程），功能完整、速度偏慢——这是已知且刻意接受的取舍。

## 性能

### 输入降采样

几个模型的内部输入其实都固定在 192~320 上下，喂 1280×720 只是让「把整帧搬进 WASM」
这一步白白多搬 7 倍像素。调度器每帧按任务要求把画面等比压到长边 640 再交给模型
（`Scheduler.DEFAULT_INPUT_LONG_EDGE`）。

关键点是**这对语义没有影响**：关键点和检测框都是归一化坐标，绘制时乘回源尺寸，
所以降采样只影响模型看到的细节，不影响叠加层对齐。人脸识别是唯一的例外——它要抠
112×112 的裁剪送去 SFace，源画面越清楚越好，所以刻意吃原始分辨率。

### 每帧的推理预算

以前是「所有到点的任务在同一帧里排队跑完」，于是各任务耗时直接相加：人脸 20ms +
姿态 46ms + 检测 122ms + 分类 54ms ≈ 240ms/帧，全开只有 3fps。

现在调度器每帧最多花 `inferenceBudgetMs`（16ms）去**开新任务**，再用一个轮转游标
保证没有任务被饿死。贵的任务自然各自占一帧，便宜的任务一帧能跑好几个。
配套地，每个任务有 `minIntervalMs` 默认值（人脸/姿态/识别 66ms，物体检测/分类 250ms），
**画面照常满帧画，结果按各自的节奏更新**。静态图片不走节流，一次性算完。

### 实测（本机无头 Chromium，CPU(WASM) 后端）

单任务耗时：人脸关键点 ~35ms、姿态 ~45ms、物体检测 ~130ms、图像分类 ~60ms、
人脸检测 ~125ms、人脸识别 ~100ms。**物体检测是绝对的瓶颈**，而且从 1280 降到 640
几乎没有变化——它的成本在模型本身而不是输入搬运，分类同理。

连续 10 分钟长跑（人脸关键点 + 姿态 + 物体检测三个任务同开）：

| 指标 | 结果 |
|---|---|
| 渲染帧率 | 稳定段 **14~15 fps**（前几分钟受同机其它进程干扰会掉到 6~10） |
| JS 堆 | t+20s 237MB → t+600s 244MB，**中间 9 分钟稳定在 244~245MB，无持续增长** |
| DOM 节点数 | 207 → 207，无泄漏 |
| 隐藏标签页 6s 再回来 | 不崩，帧率恢复正常 |

> ⚠️ 这组数字是**最差情况**：无头浏览器走软件渲染（SwiftShader），MediaPipe
> 的 `pickDelegate()` 探测到软件渲染后强制退回 CPU(WASM) 后端
> （性能面板顶栏会显示 `后端 WASM (CPU)`）。真实机器上带硬件 WebGL 的浏览器会走
> GPU delegate，这几个模型通常快 5~10 倍。想让 CPU 路径也快一档，把 COOP/COEP
> 配上就行（`crossOriginIsolated` 为真时 ORT 会用 4 线程）。

## 人脸识别怎么用

1. 勾选「人脸识别 1:N」，UI 会自动带上依赖的「人脸关键点」（五点对齐要用它的 478 点）。
2. 用摄像头对着某人、或在图片模式下加载一张照片，填名字点「注册当前人脸」。
3. 之后画面里的每张脸都会和库里比一遍，命中显示绿色名字 + 相似度，未命中显示灰色的
   `未知 NN.N%`。

「判定阈值」滑杆默认 `0.363`（OpenCV SFace 的推荐余弦阈值），调它能直接看到误识/漏识的
权衡。实测（注册 Obama / Biden / Merkel，用另一张照片和旋转 15° 的版本回放）：

| 阈值 | Obama（99.0%） | Biden（56.6%） | 陌生人 lena（19.0%） |
|---|---|---|---|
| 0.70 | ✅ | ❌ 漏识 | ✅ 正确拒绝 |
| 0.50 | ✅ | ✅ | ✅ |
| 0.363 | ✅ | ✅ | ✅ |
| 0.25 | ✅ | ✅ | ✅ |
| 0.19 | ✅ | ✅ | ❌ **误识**成 Biden |

人脸特征存在浏览器 **IndexedDB** 里（明文向量）。这是本地演示用途，**真实身份信息
请自行评估合规风险**。

## 模型来源与许可证

对外开源、不商用，所以只选许可证干净的组件。**没有引入 Ultralytics YOLO 全系
（AGPL-3.0），也没有用 InsightFace 的官方权重（非商用研究许可）**——姿态因此只用
MediaPipe，人脸识别因此改用 SFace。

| 组件 | 版本 | 许可证 | 来源 |
|---|---|---|---|
| `@mediapipe/tasks-vision` | 1.0.1 | Apache-2.0 | npm |
| `onnxruntime-web` | 1.29.0 | MIT | npm |
| Vite | 8.x | MIT | npm |
| `blaze_face_short_range.tflite` | — | Apache-2.0 | mediapipe-models |
| `face_landmarker.task` | — | Apache-2.0 | mediapipe-models |
| `pose_landmarker_{lite,full}.task` | — | Apache-2.0 | mediapipe-models |
| `efficientdet_lite0.tflite` | — | Apache-2.0 | mediapipe-models |
| `efficientnet_lite0.tflite` | — | Apache-2.0 | mediapipe-models |
| `face_recognition_sface_2021dec.onnx` | — | Apache-2.0 | opencv_zoo |

MediaPipe 的 WASM 运行时随 `tasks-vision` 的 npm 包一起分发，同样是 Apache-2.0。

### 资产体积

| 目录 | 体积 | 说明 |
|---|---|---|
| `public/models/mediapipe/` | 43 MB | 6 个 `.task` / `.tflite` |
| `public/models/onnx/` | 37 MB | SFace，最大单文件 |
| `public/wasm/mediapipe/` | 23 MB | SIMD + nosimd 两套运行时 |
| `public/wasm/ort/` | 14 MB | 纯 WASM 版胶水层 + wasm |
| `dist/` 合计 | 128 MB | 其中 JS/CSS 只有 684 KB |

懒加载下用户实际只会下载勾选过的任务所需的模型，SFace 那 37MB 只有开人脸识别才会走网络。
模型用 **Cache Storage** 持久化（`cv-models-v1`），二次访问不走网络。

## 换成自己的模型

1. 把文件放到 `public/models/...`，或改 `scripts/fetch-assets.mjs` 的清单。
2. 在 `src/tasks/<你的任务>.ts` 里实现 `VisionTask` 接口（`src/tasks/types.ts`）：

```ts
readonly id, name, stage, hint
readonly assets: AssetSpec[]        // 懒加载清单，驱动下载进度
readonly options: TaskOptionSpec[]  // UI 上自动生成滑杆 / 下拉
readonly minIntervalMs: number      // 建议的重算间隔
infer(ctx: FrameCtx): void | Promise<void>   // 只算
draw(overlay: Overlay): void                 // 只画
```

3. 在 `src/tasks/registry.ts` 里注册。

`ctx.element` 是调度器降采样后的画面（喂模型用），`ctx.source.element` 是原始分辨率；
`ctx.shared` 用于任务之间传中间结果（人脸识别就是这么拿到关键点的 478 点的）。

## 目录结构

```
src/
├── main.ts              装配入口
├── ui/                  layout / controls / stats / gallery
├── source/              FrameSource / CameraSource / ImageSource
├── render/Overlay.ts    坐标映射 + 画框/画点/画骨架/画标签
├── tasks/               每个算法一个文件 + types.ts / registry.ts
├── face/                FaceGallery(IndexedDB) / match.ts(余弦匹配)
├── runtime/             mediapipe / ort / scheduler / assets
└── util/                imageOps(对齐) / throttler / storage
```

## 已知限制

- **多人姿态不做**。当前只支持单人 33 点。将来要做的话用 RTMPose（Apache-2.0），
  不要用 YOLO-Pose（AGPL）。
- **人脸识别对镜像脸不敏感**。ArcFace 家族的模型不是镜像不变的，左右翻转后的
  相似度会掉；正脸、换照片、小角度旋转（±15° 实测 96%+）都没问题。
- **MediaPipe 的多线程拿不到**（见上文），要么接受单线程，要么把任务搬进 Worker。
- **`public/models/` 与 `public/wasm/` 没有进版本库**（116MB 二进制会让仓库永久变重）。
  `git clone` 之后必须先跑一次 `pnpm fetch-assets` 才有模型。
  `gh-pages` 分支上的 `dist/` 是带全量模型的，所以 **Pages 部署和离线分发不受影响**。
- 人脸库是 IndexedDB 里的明文特征向量，仅适合本地演示。

## 还没做（评估过的）

- **把推理挪进 Web Worker**。这是唯一能同时解决「MediaPipe 多线程」和「重任务阻塞
  主线程」的路子：Worker 里用 `vision_wasm_module_internal`（就是官方为 Worker 准备的
  ESM 变体），主线程只管 `createImageBitmap(video)` 后 transfer 过去，渲染循环就不会被
  115ms 的物体检测拖住。代价是五个任务文件 + 调度器都要改成消息驱动，改动面较大，
  所以这一轮先靠降采样 + 每帧预算把问题压到了可接受范围。
- **SFace FP16 量化**：试过了，**这条路当前走不通**，别再重复。
  `onnxconverter_common.float16.convert_float_to_float16()` 能把文件从 38.7MB 压到
  24.3MB（62.7%，还不到一半，因为 `keep_io_types=True` 会补一堆 Cast），但产物
  ONNX Runtime Web 直接拒绝加载：

  ```
  Type Error: Data in initializer 'bn1_moving_var' has element type tensor(float16)
              but usage of initializer in graph expects tensor(float)
  ```

  原因是这个 graph（opset 11）有 29 个 `BatchNormalization`，转换器把它们的
  initializer 变成 fp16 却让节点本身留在 fp32；把 `BatchNormalization` /
  `PRelu` / `Dropout` 加进 `op_block_list` 也一样。要真做得把 BN 手工展开成
  fp32 的等价算子链再转换。
- **顺手一提**：SFace 的 37MB 里大部分是 MobileFaceNet 主干权重，就算量化成功,
  WASM 后端也没有原生 fp16 运算，省下的主要是**下载体积**而不是推理时间——
  这条路的性价比本来就不高。
