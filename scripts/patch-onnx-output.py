#!/usr/bin/env python3
"""
把 timm 导出的 MobileNetV3 ONNX 的头砍掉，改成输出**分类头之前的池化特征**。

为什么必须这么做：timm/huggingface 导出的这份图只有 `logits` 这一个输出，
即 1000 维的 ImageNet 分类分数。拿它当特征向量去比掌纹是很差的——
那 1000 维是 ImageNet 的语义方向，跟掌纹褶皱这种纹理信息基本无关。

图里 `/act2/Mul_output_0` 是分类头（Gemm）之前的 1024 维特征，也就是
timm 里 `forward_head(pre_logits=True)` 拿到的东西，才是标准的「倒数第二层特征」。
它已经存在于图中，只是没被声明成图输出，所以这里补一条声明就行——
**不改任何算子、不改任何权重**，只是多暴露一个出口。

用法：
    python3 scripts/patch-onnx-output.py <输入.onnx> <输出.onnx>
"""
import sys

import onnx

SRC, DST = sys.argv[1], sys.argv[2]
TAP = "/act2/Mul_output_0"

model = onnx.load(SRC)
graph = model.graph

if any(o.name == TAP for o in graph.output):
    print(f"  skip   {DST} 已经是带特征输出的版本")
    sys.exit(0)

# 确认这个张量真的存在，别把打错字变成静默的坏产物
producers = {out for node in graph.node for out in node.output}
if TAP not in producers:
    print(f"✗ 图里找不到张量 {TAP}，模型结构可能变了，请重新确认要暴露的层", file=sys.stderr)
    sys.exit(1)

# 先把原来那些由分类头产生的图输出声明删掉，否则图里会留下「声明了但没有节点产出」的悬空输出
head_outputs = {"logits"}
keep_out = [o for o in graph.output if o.name not in head_outputs]
graph.ClearField("output")
graph.output.extend(keep_out)

graph.output.append(onnx.helper.make_tensor_value_info(TAP, onnx.TensorProto.FLOAT, ["batch_size", 1024, 1, 1]))

# 顺手删掉分类器：要暴露的 /act2/Mul_output_0 是它的**输入**，
# 所以只砍它自己和它后面的 Flatten，前面的 conv_head / act2 都得留着。
HEAD_NODES = {"/flatten/Flatten", "/classifier/Gemm"}
keep = [n for n in graph.node if n.name not in HEAD_NODES]
removed = len(graph.node) - len(keep)
graph.ClearField("node")
graph.node.extend(keep)

# 分类头的权重也一并清掉，否则依然算在文件体积里
USED = {i for n in graph.node for i in n.input}
head_init = [i.name for i in graph.initializer if i.name not in USED]
for name in head_init:
    init = next(i for i in graph.initializer if i.name == name)
    graph.initializer.remove(init)

onnx.checker.check_model(model)
onnx.save(model, DST)
print(f"  patch  {DST}（砍掉 {removed} 个分类头节点、{len(head_init)} 个权重张量，改为输出 {TAP}）")
