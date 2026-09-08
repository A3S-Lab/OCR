<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="A3S OCR validates a bounded image, routes it through an explicit provider, and returns recognized text with canonical source evidence">
</p>


<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <strong>面向 Rust 和 A3S 的提供者 OCR，结果中保留源出处。</strong>
</p>

<p align="center">
  <a href="https://github.com/A3S-Lab/OCR/actions/workflows/ci.yml"><img alt="CI status" src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/OCR/ci.yml?branch=main&amp;style=flat-square&amp;label=CI"></a>
  <a href="https://github.com/A3S-Lab/OCR/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/A3S-Lab/OCR?display_name=tag&amp;sort=semver&amp;style=flat-square&amp;color=2864e8"></a>
  <a href="https://crates.io/crates/a3s-use-ocr"><img alt="a3s-use-ocr on crates.io" src="https://img.shields.io/crates/v/a3s-use-ocr?style=flat-square&amp;color=5420bd"></a>
  <a href="https://docs.rs/a3s-use-ocr"><img alt="docs.rs documentation" src="https://img.shields.io/docsrs/a3s-use-ocr?style=flat-square"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust 1.82 or newer" src="https://img.shields.io/badge/Rust-1.82%2B-a4a8b2?style=flat-square"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-17181a?style=flat-square"></a>
</p>

<p align="center">
  <a href="#quick-start">快速开始</a> ·
  <a href="#responsibility-boundary">边界</a> ·
  <a href="#result-contract-ocr-plus-provenance">合同</a> ·
  <a href="#providers">提供商</a> ·
  <a href="ROADMAP.md">路线图</a> ·
  <a href="#cli-and-mcp-surfaces">CLI &amp; MCP</a> ·
  <a href="#development">开发</a>
</p>

---

`a3s-use-ocr`是内置的背后独立维护的OCR库
A3S Use OCR 路线。其稳定边界为[`OcrProvider`](#the-provider-interface)，
不是一个单一的模型。

每次提取都从相同的客户端拥有的工作开始：解析有界的
本地图像，验证其媒体类型，读取一次，并计算规范源
证据。只有这样，字节才会传递给注入的提供程序。供应商
必须声明其源传输策略并且不能替换源路径，
media type, size, or SHA-256 recorded by `OcrClient`. Both built-in providers
reuse A3S Power's embedded, model-neutral inference substrate; neither enables
Power's HTTP server or opens its own listener.

## Responsibility boundary

A3S OCR 识别一幅有界图像并返回 OCR 证据。它不是一个
document parser.

| A3S OCR owns | Delegated to A3S Power | Outside this repository |
| --- | --- | --- |
| PP-OCRv6 和 Unlimited-OCR 拓扑、资源、预处理、解码、标签、置信度和源像素几何 |类型设备、准入、权重完整性和驻留、取消、私人遥测、TEE 兼容控制和执行收据 | Office/PDF 页面库存、渲染、跨页面层次结构、证据协调、代理规划和文档检查点 |

PDF rasterization and Office parsing belong to their owning components. A
document-level consumer such as A3S Parser may preserve `OcrResult` blocks and
较大图表内的收据，但不得将 OCR 模型所有权移至
解析器。 Power 保持模型中立，不包含 OCR 架构或
asset.

## Quick start

安装A3S Use后，在阅读之前检查配置的提供程序
image:

~~~bash
a3s use ocr doctor --json
~~~

诊断报告提供者、引擎、型号、准备情况和
`sendsSourceOffDevice`政策。对于默认的本地提供商，安装固定的
如果诊断结果表明模型包，则提取：

~~~bash
a3s install use/ocr
a3s use ocr extract ./scan.png --json
~~~

独立的二进制文件公开相同的域操作：

~~~bash
a3s-use-ocr doctor --json
a3s-use-ocr extract ./scan.png --json
a3s-use-ocr serve --mcp
~~~

### 将客户端嵌入 Rust

默认功能集包括 PP-OCRv6、MCP 和 CLI：

~~~bash
cargo add a3s-use-ocr
~~~

~~~rust
use a3s_use_ocr::{OcrClient, OcrRequest, UseResult};

async fn extract(path: impl Into<std::path::PathBuf>) -> UseResult<String> {
    let client = OcrClient::from_env()?;
    let result = client.extract(OcrRequest { path: path.into() }).await?;
    Ok(result.text)
}
~~~

当应用程序只需要中性时使用`default-features = false`
合同和客户。

## 分阶段批量提取

`OcrClient::extract_batch` 接受稳定的调用者拥有的槽 ID 和类型化的
舞台布景。它总是按调用者顺序返回槽，即使在源验证时也是如此
或者一个提供者阶段失败：

~~~rust
use a3s_use_ocr::{
    OcrBatchRequest, OcrBatchSlotId, OcrBatchSlotRequest, OcrClient, OcrStage,
    UseResult,
};

async fn extract_surfaces(client: &OcrClient) -> UseResult<()> {
    let request = OcrBatchRequest::new(
        vec![OcrStage::Preprocessing, OcrStage::Text],
        vec![
            OcrBatchSlotRequest::new(OcrBatchSlotId::new("slide:1")?, "slide-1.png"),
            OcrBatchSlotRequest::new(OcrBatchSlotId::new("slide:2")?, "slide-2.png"),
        ],
    )?;
    let result = client.extract_batch(request).await?;
    assert_eq!(result.slots[0].slot_id.as_str(), "slide:1");
    Ok(())
}
~~~

提供者中立的阶段词汇是定向、预处理、布局、
文本、表格、公式和印章。提供者描述符声明它的子集
可以完成；未实现的阶段返回为 `unsupported`，从不
从文本推断。现有提供商的兼容性适配器支持
只有文字阶段。 PP-OCRv6 目前声明了预处理和文本，其中
预处理意味着有界图像解码和规范化。目前还没有
索赔表或密封检测。分别建造的
`DocumentFastOcrProvider` 将该文本提供者与固定的
SLANet-Plus 有线表格模型并声明预处理、文本和表格。当
配置单独固定的 PicoDet 布局包，同样显式
提供商还声明密封检测。它仍然是选择加入的，因为每个添加的
模型及其局限性必须对主机保持可见。

分阶段批处理模式 v2 要求每个已完成的表或密封阶段都携带一个
确切的源图像像素画布上的有界类型有效负载。表证据
保留检测到的表格区域、可选网格尺寸、合并单元格
跨度、文本以及仅由提供者实际提供的单元格几何形状。密封件
证据保留其确切区域、可选识别、规范画布
可见标记被剪掉时的边缘，以及模型是否确认了
该页面上的对象或仅保留一个`boundary-candidate`。客户拒绝
无效的多边形封套、画布区域外、重叠或网格外
之前的单元格、伪造的剪辑、重复的身份和无界文本
发布结果。
跨页表和骑手密封协调仍然是解析器的职责；
OCR 生成页面本地证据，并且从不连接页面本身。

### 选择加入有线餐桌提供商

将 `A3S_OCR_SLANET_PLUS_MODEL_DIR` 设置为已审核的本地捆绑包，具体内容如下
库存：

~~~text
encoder/model.safetensors
slanext_wired_decoder.bin
slanext_dict_infer.txt
~~~

然后显式注入提供者：

~~~rust
use a3s_use_ocr::{DocumentFastOcrProvider, OcrClient, UseResult};

fn document_fast_client() -> UseResult<OcrClient> {
    OcrClient::with_provider(DocumentFastOcrProvider::from_env()?)
}
~~~

该提供商承认来自相交页面的保守有线表格裁剪
规则，通过A3S Power运行固定的488像素SLANet-Plus编码器，解码
局部自回归结构和单元四边形，并分配
PP-OCRv6 文本块通过源像素几何对单元进行建模。一线候选人
单独的内容永远不会作为表格证据发布。无边框桌子仍然存在
不支持，并且页面本地片段不会被标记为跨页面
通过 OCR 继续。

### 选择加入模型支持的密封位置

将`A3S_OCR_PICODET_LAYOUT_MODEL_DIR`设置为审核后的本地目录
包含转换后的`model.safetensors`。签入的图表是
固定的 PaddleOCR `PicoDet-L_layout_3cls` 原始头的确定性降低；
生产环境既不加载 Paddle，也不加载 Python。该模型拥有 `seal` 类并且
主机在源像素坐标中执行有界分数过滤和 NMS。

正常的页面路径使用一个640像素的整页视图加上固定的左右
边条。审查阈值下的整页检测为`confirmed`。
低置信度边缘证据永远不会被提升：它返回为
`boundary-candidate`，必须接触声明的源画布边缘，并保持不变
如果没有下游协调，则无法作为已确认的对象发布。

对于已承认的序列，调用者可以显式地将一个槽绑定到其直接序列
前身为`with_adjacent_predecessor`。当前驱包含
有界边缘候选者，OCR 最多运行一个额外的本地视图
当前页面的边缘。这恢复了窄右边缘片段
保留两页骑手密封固定装置，而第二页独立
保留了其三个内部密封。邻接声明仅授权
额外页面-本地证据收集；解析器仍然拥有跨页匹配，
推广和规范几何。印章文字识别及通用印章
准确度分数未实施。

请求包含 1 到 256 个唯一插槽以及最多 256 MiB 的已验证数据
除了现有的每图像 32 MiB 限制之外，还增加了输入字节。畸形
请求或提供者输出形状使调用失败。源、模型加载和阶段
执行错误仍附加在其确切的槽位上，如已完成、部分、
失败、跳过或不受支持的结果。结果也符合规范
提供者和每槽模型指纹以及仅摘要执行收据；
原始源字节、张量值和本地路径不放在调度中
证据。

## 结果合约：OCR 加出处

提供者拥有认可。 `OcrClient`拥有证据信封。

|属于`OcrClient` |归提供商所有 |
| ---| ---|
|规范路径、检测到的媒体类型、字节大小、SHA-256 |识别文字和模型身份|
|输入范围和支持的图像签名 |可选置信度、类别、多边形和边界框 |
|提供者输出验证 |准备情况消息和特定于提供商的警告 |
|最终`OcrResult`组装|宣布设备外源政策 |

本机结果还可能包含 `executionReceipts`。每张收据均绑定一个
型号系列和修订版、精确的重量摘要、功率运行时间/设备
身份和规范输入/输出摘要。下游解析器应该
保留这些收据和 OCR 证据。

稳定的结果形状使源保持在 OCR 证据旁边：

~~~jsonc
{
  "provider": "unlimited-ocr",
  "engine": "a3s-power-native",
  "model": "baidu/Unlimited-OCR",
  "source": {
    "path": "/canonical/path/to/scan.png",
    "mediaType": "image/png",
    "size": 12345,
    "sha256": "..."
  },
  "text": "...",
  "blocks": [
    {
      "page": 1,
      "text": "...",
      "category": {"rawLabel": "title", "role": "title"},
      "boundingBox": {"x": 12, "y": 24, "width": 208, "height": 74},
      "boundingBoxes": [
        {"x": 12, "y": 24, "width": 208, "height": 34},
        {"x": 12, "y": 64, "width": 180, "height": 34}
      ]
    }
  ],
  "executionReceipts": [
    {
      "schema": "a3s.power.embedded-execution-receipt.v1",
      "model": {"family": "baidu/Unlimited-OCR", "revision": "07dea832...", "weightsSha256": "..."},
      "runtime": {"name": "a3s-power-native", "version": "0.8.0", "device": "metal:0"},
      "input": {"representation": "image-request", "sha256": "...", "byteLength": 12345, "itemCount": 1},
      "output": {"representation": "utf8-text", "sha256": "...", "byteLength": 321, "itemCount": 287}
    }
  ],
  "warnings": []
}
~~~

类别、置信度和几何形状是可选的。 `category.rawLabel` 保留
有界的提供者标签，但不声明提供者分类已关闭；
`category.role` 是一种保守的提供商中立解释。组件
盒子保留精确的提供者几何形状，而`boundingBox`是它们的兼容性
信封。 OCR 输出是源自来源的证据，而非经过验证的来源
文本。

## 提供商

提供者选择是一个类型化的对象，而不是原始的后端名称开关。

|供应商| OCR 拥有的实施 |执行基板|源边界|
| ---| ---| ---| ---|
| `PpOcrV6Provider` |检测/识别图、图像管道、DB/CTC 后处理 |嵌入式A3S Power |设备始终在线 |
| `DocumentFastOcrProvider` | PP-OCRv6 文本、SLANet-Plus 有线表格结构和可选的 PicoDet-L 密封位置以及键入的候选边界 |嵌入式A3S Power |设备始终在线 |
| `UnlimitedOcrProvider` |视觉塔、投影仪、解码器、分词器、生成和接地 |嵌入式A3S Power |设备始终在线 |
|定制`OcrProvider` |由实现定义 |由实现定义 |在其描述符中必需 |

### 默认：PP-OCRv6

默认 A3S 集成使用：

- 提供商 ID：`pp-ocr-v6`
- 发动机：`a3s-power-native`
- 固定捆绑包：`PP-OCRv6_small`
- 接送政策：仅限本地

它的管道是明确的：

~~~text
bounded decode → cross-image letterbox → batched detection → per-slot DB
               → identity-bound crop plans → stable width sort
               → cross-image crop batches → CTC decode → ordered evidence
~~~

OCR 拥有的发布包固定检测和识别 SafeTensors
加上他们的推理配置。安装验证存档长度
和 SHA-256，仅提取四个声明的文件，并记录确切的 Power
体重消化。嵌入式 SLANet-Plus 和 PicoDet 图身份散列
存储库的 LF 标准化 JSON blob； `.gitattributes` 和摘要测试拒绝
平台线路末端漂移。安装和维修仍然明确：

~~~bash
a3s install use/ocr
a3s install use/ocr --force
~~~

模型下载绑定连接设置并停止读取，而无需强加
总传输截止日期，因此健康的慢速链接仍然可以完成固定的
存档。中断的主体从精确的验证字节范围重试；一台服务器
忽略范围会重新启动暂存文件而不是追加。的
相同的有限重试预算涵盖瞬时连接和源故障。的
完整的存档仍然必须匹配其固定长度和 SHA-256 之前
激活。

`A3S_OCR_MODEL_DIR` 可以将开发构建指向显式模型包。
`A3S_USE_OCR_HOME` 覆盖用于打包、测试或
隔离安装。提供商执行经过审核的 OCR 拥有的图表计划
通过Power的共享准入、设备、限制、完整性、取消和
接收机制。它不需要 ONNX Runtime、Python、PaddlePaddle、a
子流程、推理服务或 Web 侦听器。

分阶段 PP-OCRv6 批次重用精确的、延迟加载的 Power 模型会话，并且
从实时主机/设备内存中规划确定性的连续微批次
快照。每个承认的微批次持有一个取消令牌、设备
允许，引擎锁定其插槽并发出 schema-v4 收据
会话声明、计划摘要、批次索引/计数、槽计数和队列
证据。检测预处理和DB后处理最多使用16个有界
工人并保留准确的槽顺序。快速探测器的边界最长
边长为 896 像素，而多边形则映射回不可变源，并且
识别裁剪原始图像。源上的空快速结果
至少 32 个级别的信道变化接收一次标量质量重试
最大边长 4,000 像素；两张检测收据仍附在其中。这次重试
保护空结果质量，但不能保证防止部分小文本
错过了。

调整大小后的不同尺寸的图像在左上角以信箱形式显示
当每个槽保留至少 90% 画布填充时，一张归一化黑色画布。
OCR 确定性地分割低填充形状异常值和任何其
审查的峰值中间值将超过 Power 的张量元素限制。每个
兼容队列最多包含 16 个图像并执行一个动态
`[B,3,H,W]` 检测图调用。电源验证引导轴装配和
输出分区； OCR 保留每个槽的内容范围，不包括填充
来自数据库后处理，并将通过该范围的多边形映射到源中
像素。然后，OCR 将已承认的图像中检测到的作物展平，同时
保留准确的槽位、检测和读取顺序身份。它稳定排序
动态识别宽度为最多八种作物的规范组，其
最宽的画布比最窄的画布宽不超过 16 像素。因为每一个
识别画布宽度至少为 320 像素，审核范围最多添加
5% 右填充，同时折叠像素级裁剪抖动；更大的宽度
差异仍然是分开的。仅当以下情况时才合并相邻规范组
它们的最终画布宽度已经相同，具有严格的物理限制
32种农作物。这不会改变填充或模型输入值。视角
作物和识别张量使用共享的 Rayon 工作池并恢复
相同的确定性顺序；标量与批量张量测试是字节精确的。的
规划器仅物化活动组并将块和收据恢复到
他们的源插槽。无限宽度混合仍然被禁止，因为
PP-OCRv6 识别具有全局宽度上下文，可以更改解码的文本。一个
失败的共享图调用通过标量路径重试受影响的作物，因此
非取消失败仍然是孤立的；取消仍然会终止
承认的请求。仅包含空格的识别结果将被忽略
来自公共区块，而不是发布无效的空证据；这个过滤器
在推理之后运行，不是检测器置信度的捷径。

有界宽度释放门使用 SHA 固定的解析器栅格。三页的
跨页表固定装置保留其精确的文本指纹，`6x6/29`，
`8x7/25`和`3x6/17`网格/单元，以及两个延续边。两页的
骑手密封固定装置保留其精确的文本指纹，三个完整的密封，
两个 IoU 检查的右边缘片段和一个连续恒等式。的
全文档门将其扩展到所有六个表格页面和所有 29 个骑手印章
页数：CUDA 保留精确的文本和结构化几何指纹，71 个表
单元格、两个表延续、12 个完整密封和两个协调边界
没有未解决的候选者的片段。在命名为RTX 4090的开发上，
等画布合并加并行识别预处理首先减少
29 页 CUDA 中值从 8.400 秒到 6.834 秒。当前字节精确 GELU
然后融合将五轮交替顺序中位数从 1.489 减少到 1.463
六页表格文档（4.101 页/秒）从 6.255 秒到 5.838
29 页骑手印章文档的秒数（4.968 页/秒）。随后的
当前机器负载下使用的channel-bias-fusion A/B 九个交替
桌门运行次数和密封门运行次数为 5 次：中位数从 1.387 下降到
1.340 秒（4.326 至 4.478 页/秒，延迟降低 3.4%）以及从 6.067 至
分别为 5.960 秒（4.780 至 4.866 页/秒，延迟降低 1.8%）。每个
运行保留了相同的文本、表格延续、单元格、密封位置和
边界片段断言。随后使用的 LayerNorm-affine-tail A/B
相同的交替协议：九轮表的中位数从 1.270 下降到
1.215 秒（4.724 至 4.938 页/秒，延迟降低 4.3%），而五次运行
海豹中位数实际上持平于 5.848 秒与 5.840 秒（4.959 秒与
4.966 页/秒）。当前单次运行 CPU 捕获次数为 46.334
分别为 334.596 秒（0.129 页/秒）和 334.596 秒（0.087 页/秒）。的
CUDA 结果仍然低于 10 页/秒的完整精细解析目标。这些是
特定于装置的正确性和延迟诊断，而不是语料库范围的 OCR
准确性或吞吐量声明。

识别不再具体化完整的 18,710 类概率行
在主机上。 OCR 在 Power 上应用确定性模型拥有的投影
执行设备和传输`[class index, score, source-finite marker]`
每个 CTC 时间步长。反轴减少保留了标量解码器的
最后一类平局规则，而标记覆盖每个源概率而不是
不仅仅是选定的分数。对于经过审查的 `[1,40,18710]` 输出，此
将主机实现和收据哈希从 2,993,600 字节减少到 480
字节 (**6,236.7x**)。投影修订是模型/会话的一部分
执行身份，并且执行收据承诺准确的预计
CTC 解码消耗的张量。

固定的 Power 运行时还融合了经过审查的 CUDA 乘法器一深度方向
卷积：17 个检测层和 14 个识别层现在执行一个
每个节点的 F32 内核，而不是每个内核一个设备范围的乘法/加法序列
位置。检测偏差在最后一项之后应用于同一内核中。
显式舍入到最近的算术保留了先前的累积顺序，并且
Power 的选定设备奇偶校验门是字节精确的； CPU 和不支持的张量
布局保留其现有路径。 OCR 仍然拥有图形库存并且
端到端输出奇偶校验。

相同的固定电源版本私下熔断相邻的单消费者 F32
CUDA 上的`HardSigmoid`-to-`Mul` 通道门。经审查的 OCR 计划包含 13
此类检测位点和五个识别位点；图库存测试锁
这些很重要。每个在 `[N, C, H, W]` 上匹配的 `[N, C, 1, 1]` 门都将替换
原始四次激活传递加上广播乘法一
字节精确的内核，每个删除四个启动和四个中间缓冲区
网站。图拓扑、收据和 OCR 所有权不会改变。 CPU 和每个
未经审查的数据类型、形状、广播形式或布局逐节点保留
执行。

识别还包含 13 个相邻的、单消费者分解的 GELU 链，
每个都表示为 `Div`-`Erf`-`Add`-`Mul`-`Mul`，具有三个标量初始值设定项。
固定的 Power 执行器在模型加载时捕获这些标量并运行
每个链作为一个 CUDA 内核，具有显式除法、加法和
乘法舍入边界。它的字节精确内核和完整的图门
删除每个链的四次启动，而不重写 OCR 拥有的图； 138 号
保留的 29 页密封门中的识别调用因此避免了 7,176
发射。 CPU 和每个不匹配的图形、数据类型、布局、设备、输出或
共享中间保留普通的逐节点执行。

识别图还包含 10 个 `Conv`-bias-ReLU 前缀，13 个
`Conv`-bias 前缀为这些 GELU 链提供数据，以及五个 `Conv`-bias 前缀
喂养门控 HardSigmoid 相乘。 OCR 拥有的拓扑测试锁定 28
计数，针对卷积输出通道的精确 F32 `[1,C,1,1]` 偏差形状，
身份深度和私人消费者关系。固定的 Power 执行器
将每个卷积保留在其现有后端并折叠通道添加
进入以下字节精确的 CUDA 激活。每次识别呼叫可避免 28
更多的全张量发射和缓冲；保留的138调用封门避免
3,864。快速测量需要发射限制的 32 位通道索引
路径。 CPU 和每个未经审查的偏差、形状、拓扑、设备、数据类型或布局
保留普通图。

识别中五个分解的最后轴 LayerNorm 块保留其两个
均值减少、居中和平方，而 Power 则私下融合每个
精确的`Add(epsilon)`-`Sqrt`-`Div`-`Mul(scale)`-`Add(bias)`尾部。 OCR 拥有
库存测试锁定五个相邻的私有窗口、标量 epsilon 和
120 个元素的缩放/偏差初始值设定项。显式的 F32 舍入边界使得
CUDA 尾部字节与原始五个节点完全相同。每次识别呼叫都避免
20 个进一步的发射和中间缓冲区；保留的138调用封门
避免 2,760。 CPU 和每个未经审查的拓扑、形状、设备、数据类型或布局
保留普通执行。

当前的质量证据涵盖固定的 30 块官方图像和
以 144 DPI 渲染的清晰 8 点和 12 点 PDF 文本。五点合成
文本未通过确切的发布，并且不是受支持的质量声明。的
命名硬件解析器集成门由消费解析器记录；
该箱子不会将该工作负载转化为通用 OCR 吞吐量声明。

Linux CI 安装确切的固定捆绑包并在上执行两个已审查的图表
CPU。门检查规范的 Power 权重摘要、准确的输出
用于零张量检测的形状、项目计数和字节长度
识别装置。然后它会下载 PaddleOCR 的 SHA-256 固定
`general_ocr_002` 图像并执行完整的 Rust 管道：调整大小，
检测、数据库后处理、读取顺序排序、透视裁剪、批处理
识别、CTC 解码、源坐标多边形和八个 Power 执行
收据。 30 个输出块根据使用生成的参考进行检查
使用显式文本、分数和四点的 Paddle 3.3.1 和 PaddleOCR 3.7.0
坐标公差。同一门比较标量和的一种官方作物
跨图像批量宽度为 2，需要相同的文本和几何形状、识别
`0.00001` 内的信心，一张共享认可收据，以及精确的 2x
输入张量大小。 Paddle、Python 和 ONNX 运行时不是测试或运行时
这个crate的依赖项。

`a3s-use-ocr-execution-bench` 添加了严格的、无路径的真实提供商基准
对于那个固定的图像。它将第一个惰性模型会话与温暖模型会话分开
执行，每毫秒采样处理 RSS，保留检测和
识别电源指纹，并抑制输出漂移。固定的对象是
名为 `.png` 上游，但具有 JPEG 字节签名；来源证据如下
字节。调试或修改树报告仅用于诊断。参见
[PP-OCRv6 Execution Baseline Protocol](docs/execution-baseline.md) 对于
发布程序和索赔边界。

有关电源/OCR，请参阅[Native Inference Architecture](docs/native-inference.md)
所有权边界、模型转换和安装完整性、执行收据、
和 TEE/隐私发布门。对齐后请参见[`ROADMAP.md`](ROADMAP.md)
Power/OCR/Parser 交付序列和 TurboOCR 派生的工作流。

### 可选：baidu/Unlimited-OCR

启用`unlimited-ocr`功能来运行经过审查的3B视觉语言模型
处理中。 A3S OCR 拥有原生 Rust 模型拓扑、分词器、
预处理、生成循环、修订引脚和基础解析器。 A3S Power
提供共享设备、准入、权重完整性、驻留、路由、
取消、遥测和接收机制。

~~~rust
use a3s_use_ocr::{
    OcrClient, ResidencyBudgetPolicy, UnlimitedOcrConfig,
    UnlimitedOcrProvider,
};

fn local_unlimited_ocr() -> Result<OcrClient, Box<dyn std::error::Error>> {
    let residency = ResidencyBudgetPolicy::new(5_000, 5_000)?
        .with_host_reserve_bytes(2 * 1024 * 1024 * 1024)
        .with_device_reserve_bytes(512 * 1024 * 1024);
    let config = UnlimitedOcrConfig::new("/models/baidu-unlimited-ocr")?
        .with_residency_budget_policy(residency)?
        .with_max_generated_tokens(8_192)?;
    OcrClient::with_provider(UnlimitedOcrProvider::new(config)?)
}
~~~

`UnlimitedOcrConfig::from_env` 读取相同的本地路径
`A3S_UNLIMITED_OCR_MODEL_DIR`。提供者的创建是惰性的：它不执行任何模型
下载、进程启动、网络请求或套接字绑定。会话加载
仅接受固定的上游修订版本
`07dea832e22aefee32ad281d4b80551282e1c168`，验证确切的分词器并
处理器资产，并要求 Power 执行单个完整的 SafeTensors 哈希
和库存验证路径，包括任何明确配置的验证
复制品。审核后的主权重文件正好是 6,672,547,120 字节
使用 SHA-256
`2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6`。
不可跳过的官方库存门仅解决修订问题
`07dea832e22aefee32ad281d4b80551282e1c168`，验证 Hugging Face 的存储库
提交加上链接文件大小和 SHA-256，以及范围读取 334,632 字节
SafeTensors JSON 标头，而不是下载 6.7 GiB 有效负载。它检查
固定的小资产摘要、官方指数、所有 2,710 个 BF16 张量名称，
形状和字节范围、精确的 6,672,212,480 字节张量有效负载布局，以及
OCR 拥有的规范库存摘要。会话负载与 Power 的比较
在推理之前使用相同的摘要完全散列库存。这个门证明
检查点身份和拓扑。一个单独的本地数字门执行
完成官方检查点并保持模型输出验收独立
从库存验收。

数字门下载现有的 SHA-256 引脚 PaddleOCR 登机
通过图像，在 Rust 中导出固定的 640×528 无损裁剪，并获得全部 64 分
通过相同的 KV 缓存、无重复和解码器的上游 CPU 引用令牌
生产生成使用的循环。它记录了每个预期的代币排名和
logit delta，然后执行第二次自由运行的贪婪解码。苹果CPU
Accelerate 完全匹配所有 64 个参考标记。金属保留第一
正好是 15，最多有两个 2 阶边界，最大 logit 为 0.25
三角洲；明显的区别是一个可选的前导标点符号和一个
三像素标题框边缘。两条路径必须返回相同的三个`header`，
`title` 和 `text` 块、审阅的文本以及其中的源像素几何图形
三像素边界。审核时设置`A3S_UNLIMITED_OCR_REQUIRE_EXACT_PARITY=1`
预计将提供完全由教师强制的令牌平等的后端。

本机转发路径遵循权威的上游实现：

~~~text
EXIF-aware decode
  → 1024px global view + optional bounded 640px tile grid
  → SAM ViT-B detail tower
  → CLIP-L semantic tower over SAM patch features
  → 2048 → 1280 projector + spatial newline/view separator packing
  → 12-layer DeepSeek-style decoder (64 routed experts, exact top-6)
  → deterministic greedy decode + sliding no-repeat 35-gram
  → bounded Markdown and source-pixel grounding
~~~

一次逻辑提取可容纳一个电源许可和取消令牌
完整的视觉、投影仪、解码器和接地流程。路由专家
而是使用 Power 的精确批量联合和默认私有路由遥测
比第二个 OCR 本地缓存。默认情况下，缓存驻留保持为零。一个打字的，
选择加入 `ResidencyBudgetPolicy` 要求选定的 Power 运行时发现
有界主机/CUDA/金属容量并从显式导出缓存字节
分数、储备、上限和运行时间限制；金属统一内存数
一次。手动缓存字节和自动预算是互斥的。
容量快照既不会保留，也不会添加到遥测或执行中
收据。无论使用哪种显式缓存模式，有界专家预取都会重叠
共享专家计算并使用 Power 的 LFRU/LRU 布局。丢弃
等待识别未来取消该共享令牌；阻塞本机
然后工作线程停在其有界的预处理、视觉和解码器取消点。
提供商发出一份绑定源图像摘要的最终收据，并经过审核
权重收集、功率设备和用户可见的 UTF-8 文本。

CPU可用`unlimited-ocr`； `unlimited-ocr-accelerate` 支持苹果
加速 CPU 内核，同时保留 BF16 模型边界。构建与
`unlimited-ocr-metal` 对于明确的 Apple Metal 设备或
`unlimited-ocr-cuda` 用于显式 NVIDIA CUDA 设备。类型设备选择
当请求的加速器不可用时，关闭失败。跑进去
Power 的 TEE 部署保留了模型完整性、资源边界、私有性
遥测和接收保证；源字节和详细路由数据是
该提供商从未导出过。

提供者应用上游单图像提示和无重复n-gram
原生生成循环中的策略并保留生成的 Markdown。它
严格解析上游模型中审查的两种接地形式
实施：

~~~text
<|ref|>title<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>text
<|det|>text [x1, y1, x2, y2]<|/det|>text
~~~

无限 OCR 坐标使用由文档记录的封闭 `0..=999` 基础
[upstream postprocessor](https://huggingface.co/baidu/Unlimited-OCR/blob/07dea832e22aefee32ad281d4b80551282e1c168/modeling_unlimitedocr.py#L62-L111)。
A3S OCR 解析验证的输入维度并映射有效的非图像
扎根于类型化的源像素`OcrBlock`证据。每个有效组件
盒子按模型顺序保留，并且 `boundingBox` 仍然是有界并集
兼容性。有界原始标签保留在保守角色旁边：
明确的标题、标题、段落、表格、说明文字、方程式、运行
页眉/页脚、脚注、页码和代码接收匹配角色；
其他有效标签保留`unknown`而不是升级。上游
分类学被有意地视为开放的。

该实现不将模型文本评估为代码，不制造任何置信度，并且
对于缺失、畸形、超出范围、空、仅图像或
EXIF 转换接地。它从不信任生成的图像路径。这如下
上游加载程序的 EXIF 转置行为，不会发生错误标签转换
坐标为未变换的源像素。接地性能下降仍然可见
通过一个有界警告，同时保留生成的文本。诊断
验证本地资产清单，无需执行两次 6.7 GiB 哈希；的
第一个会话打开完成Power的强制完整检查点验证。

### 提供者接口

`OcrProvider` 保持对象安全，`Send + Sync`，并且独立于具体
提供商依赖项：

~~~rust
#[async_trait::async_trait]
pub trait OcrProvider: Send + Sync {
    fn descriptor(&self) -> OcrProviderDescriptor;
    fn diagnostic(&self) -> OcrProviderStatus;
    async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput>;
}
~~~

使用 `OcrClient::with_provider(provider)` 或
`OcrClient::from_provider(Arc<dyn OcrProvider>)`。描述符必须包含一个
稳定的提供商 ID、引擎名称和设备外源策略。

## CLI 和 MCP 界面

|表面|切入点|提供者行为 |
| ---| ---| ---|
| A3S Use | `a3s use ocr ...` |预留内置路由； PP-OCRv6 是当前默认值 |
|独立 CLI | `a3s-use-ocr ...` |等效的 `doctor`、`extract` 和 `serve --mcp` 操作 |
| Rust 库 | `OcrClient` |接受任何类型的提供者 |
|标准MCP| `OcrMcpServer::new(client)` |曝光 `ocr_doctor` 和 `ocr_extract` |

`OcrMcpServer` 将提供商的源转移政策投射到
`ocr_extract`工具注释。因此，定制的设备外提供商仍然存在
对 MCP 主机可见，而不是看起来像仅限本地读取。

## 功能标志

|特色|添加|
| ---| ---|
| `power-runtime` |模型中立的嵌入式A3S Power运行时；从不启用其服务器功能 |
| `ppocr-v6` |本地 PP-OCRv6 提供商、本机图形计划、安装程序、图像管道 |
| `ppocr-v6-cuda` |通过 Power 的 NVIDIA CUDA 路径进行 PP-OCRv6 和文档快速表/印章推理 |
| `benchmark` | PP-OCRv6 真实图像冷/温执行基线二进制 |
| `unlimited-ocr` | Native CPU Unlimited-OCR模型、分词器、图像管道、生成和接地 |
| `unlimited-ocr-accelerate` | Unlimited-OCR plus Apple Accelerate CPU 内核，经过审查的 BF16 操作边界 |
| `unlimited-ocr-metal` | Unlimited-OCR 加上 Power/Candle Apple Metal 设备路径 |
| `unlimited-ocr-cuda` |无限 OCR 加上 Power/Candle NVIDIA CUDA 设备路径 |
| `mcp` |提供商中立的标准 MCP 主机 |
| `cli` |独立的 CLI；组装 PP-OCRv6 和 MCP |
|默认| `ppocr-v6`、`mcp` 和 `cli` |

## 输入和信任边界

- 输入是 1 字节到 32 MiB 之间的常规本地文件。
- 支持的签名有 PNG、JPEG、WebP、GIF、BMP 和 TIFF。
- URL 和 PDF 光栅化不在当前客户合同范围内。
- 提供者无法替换由以下人员创建的规范来源证据
  `OcrClient`。
- 页从 1 开始；返回的置信度值必须是有限的并且介于 0 和
  1.
- 提供者标签和组件框列表是有界的；组件列表必须
  与其兼容性范围完全一致。
- 两个内置提供程序从不将源字节传输出设备。
- 无限 OCR 源框和类别仅从有效的、
  有界 `0..=999` 基础和解码源图像维度；畸形的
  标记永远不会成为盒子或语义声明。
- 模型安装、修复、关卡获取永不隐藏
  内部提取。
- 内置嵌入式推理边界不包含ONNX Runtime，外部
  OCR 服务、HTTP 客户端/服务器、浏览器自动化、Python 运行时、
  子进程推理或网络侦听器。

## 发展

从此crate存储库运行检查，而不是从 A3S monorepo 根运行检查：

~~~bash
cargo fmt --all -- --check
cargo test --no-default-features --lib --locked
cargo test --no-default-features --features unlimited-ocr --locked
cargo check --no-default-features --features mcp --locked
cargo test --features unlimited-ocr --locked
cargo clippy --all-targets --features unlimited-ocr --locked -- -D warnings
tools/check_official_ppocr_v6.sh /tmp/a3s-ppocr-v6-gate
tools/check_official_unlimited_ocr.sh /tmp/a3s-unlimited-ocr-gate
# With a complete reviewed checkpoint already present:
tools/check_local_unlimited_ocr_checkpoint.sh /models/baidu-unlimited-ocr
tools/check_local_unlimited_ocr_parity.sh /models/baidu-unlimited-ocr
# On macOS:
cargo check --no-default-features --features unlimited-ocr-metal --locked
cargo package --locked
~~~

该库依赖于已发布的`a3s-use-core`机器合约。 A3S Use
组装内置路由时，封装不可变的 OCR 修订版
技能和模型资产。

分阶段的 PP-OCRv6 集成固定了可发布的 A3S Power 0.8.0 修订版
`2939668e8ad38e4d3f564144d01c2a5020aa39de`。源代码构建和 CI 执行
准确的 Git 修订版。包验证还解决了声明的问题
`=0.8.0` 注册表依赖项，因此包门保持关闭状态，直到相同
电源释放在 crates.io 上可见。无路径或 `[patch.crates-io]` 覆盖
属于这个存储库。

<details>
<summary>释放所有权</summary>

该存储库拥有提供者接口，默认 PP-OCRv6 实现，
原生 Unlimited-OCR 实施、测试、模型出处、技能内容、
crate出版物和平台档案。 A3S Use拥有内置路线，选择
默认提供者、能力预测、组件策略和最终产品
装配。版本通过不可变修订和 SHA-256 绑定来满足
文物。

</details>

## 执照

根据 [MIT License](LICENSE) 获得许可。参见
[Third-Party Notices](THIRD_PARTY_NOTICES.md) 用于模型和运行时来源。
