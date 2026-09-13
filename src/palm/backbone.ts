import type { AssetSpec } from '../tasks/types'

/**
 * 掌纹特征骨干。
 *
 * 这里没有「掌纹识别模型」可用——找遍了也**没有许可宽松的掌纹预训练权重**
 * （已有的几个仓库都没带 LICENSE 文件，数据集清一色 academic-only）。
 * 所以走的是另一条路：用通用视觉骨干抽的特征做 few-shot 匹配，
 * 和人脸那边「注册几张再比对」的用法完全一致。
 *
 * 代价要说清楚：这套特征的**判别力没有经过任何公开数据集评估**，
 * 阈值是我们拿自己的手掌照片实测定的。它是功能演示，不是可用的身份认证。
 */
export interface PalmBackbone {
  id: string
  label: string
  /** 一行说明，显示在下拉框旁边 */
  hint: string
  asset: AssetSpec
  /** 特征维度 */
  dimension: number
  /**
   * 从模型输出里取出特征向量。
   *
   * 两种骨干的输出形态不同：
   * - DINOv2 吐的是 `last_hidden_state [1, 257, 384]`，第 0 行是 CLS token，
   *   也是 ViT 里约定俗成的「整图特征」；后 256 行是 16×16 的 patch token。
   * - MobileNetV3 吐出池化后的 1024 维卷积特征（图被我们改过，见
   *   `scripts/patch-onnx-output.py`）。
   */
  extract(out: Float32Array, dims: readonly number[]): Float32Array
  /**
   * 由输出形状推出特征向量的长度，必须和 `extract` 的结果一致。
   *
   * 单独列一个方法是因为「库里的向量维度对不对得上模型」要在**推理之前**就能判断
   * （换骨干时得马上告诉用户旧数据用不了了），那时候手上只有模型的 outputMetadata。
   */
  featureLength(dims: readonly number[]): number
}

/**
 * DINOv2-Small（Apache-2.0，facebook）。
 *
 * 选它是因为自监督预训练出来的特征在**纹理**上明显比 ImageNet 分类网络好用，
 * 而掌纹恰好就是纯纹理。384 维 CLS 直接当特征，不需要任何后处理。
 *
 * 用的 fp16 版本：45.5MB 对 88.5MB 的 fp32，实测两者给出的余弦相似度只差
 * 0.0001（见 tmp/probe_quant.mjs 的对照），体积却省一半。
 */
const DINOV2: PalmBackbone = {
  id: 'dinov2',
  label: 'DINOv2-Small（精细）',
  hint: 'ViT-S/14，自监督预训练，纹理特征强；45MB',
  asset: {
    url: 'models/onnx/palm_dinov2_small_fp16.onnx',
    label: '掌纹骨干 DINOv2-Small',
    bytes: 45453331,
  },
  dimension: 384,
  extract(out, dims) {
    // [1, 257, 384]：取第 0 行（CLS）
    const dim = dims[dims.length - 1]
    return out.slice(0, dim)
  },
  featureLength(dims) {
    // 只看最后一维：这是每个 token 的宽度，也就是 CLS 的长度
    return dims[dims.length - 1]
  },
}

/**
 * MobileNetV3-Small（Apache-2.0，timm 权重）。
 *
 * 轻量档：5.9MB，只有 DINOv2 的八分之一。代价是判别力弱一截——
 * 它是 ImageNet 分类网络，特征里语义成分比纹理成分重。
 *
 * ⚠️ 用的是**改过图**的版本：原始导出只有 1000 维的 `logits` 一个输出，
 * 拿分类分数当特征向量是很差的选择。`scripts/patch-onnx-output.py` 把分类头摘掉，
 * 改成输出分类前的 1024 维池化特征。**fetch-assets 会自动做这一步**，
 * 所以仓库里存的永远是打好补丁的版本。
 */
const MOBILENET: PalmBackbone = {
  id: 'mobilenetv3',
  label: 'MobileNetV3-Small（轻量）',
  hint: 'ImageNet 分类骨干，取池化特征（1024 维）；5.9MB',
  asset: {
    url: 'models/onnx/palm_mobilenetv3_features.onnx',
    label: '掌纹骨干 MobileNetV3-Small',
    bytes: 5900000,
  },
  dimension: 1024,
  extract(out) {
    // [1, 1024, 1, 1]：整段都是特征
    return out
  },
  featureLength(dims) {
    // [1, 1024, 1, 1] 后面那两维是 1，连乘正好等于 1024
    return dims.slice(1).reduce((a, b) => a * b, 1)
  },
}

export const PALM_BACKBONES: readonly PalmBackbone[] = [DINOV2, MOBILENET]

export function findBackbone(id: string): PalmBackbone {
  return PALM_BACKBONES.find((b) => b.id === id) ?? DINOV2
}
