<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="A3S OCR 校验有界图像，经显式提供者路由，并返回带规范源证据的识别文本">
</p>

<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <strong>面向 Rust 与 A3S 的提供者导向 OCR，结果中保留源出处。</strong>
</p>

<p align="center">
  <a href="https://github.com/A3S-Lab/OCR/actions/workflows/ci.yml"><img alt="CI 状态" src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/OCR/ci.yml?branch=main&amp;style=flat-square&amp;label=CI"></a>
  <a href="https://github.com/A3S-Lab/OCR/releases/latest"><img alt="最新发布" src="https://img.shields.io/github/v/release/A3S-Lab/OCR?display_name=tag&amp;sort=semver&amp;style=flat-square&amp;color=2864e8"></a>
  <a href="https://crates.io/crates/a3s-use-ocr"><img alt="crates.io 上的 a3s-use-ocr" src="https://img.shields.io/crates/v/a3s-use-ocr?style=flat-square&amp;color=5420bd"></a>
  <a href="https://docs.rs/a3s-use-ocr"><img alt="docs.rs 文档" src="https://img.shields.io/docsrs/a3s-use-ocr?style=flat-square"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust 1.82 或更新" src="https://img.shields.io/badge/Rust-1.82%2B-a4a8b2?style=flat-square"></a>
  <a href="LICENSE"><img alt="MIT 许可证" src="https://img.shields.io/badge/license-MIT-17181a?style=flat-square"></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#职责边界">边界</a> ·
  <a href="#结果契约ocr-与出处">契约</a> ·
  <a href="#提供者">提供者</a> ·
  <a href="ROADMAP.md">路线图</a> ·
  <a href="#cli-与-mcp-界面">CLI &amp; MCP</a> ·
  <a href="#开发">开发</a>
</p>

---

`a3s-use-ocr` 是内置 A3S Use OCR 路由背后独立维护的 OCR 库。其稳定边界是
[`OcrProvider`](#提供者接口)，而非单一模型。

每次抽取都以相同的客户端自有工作开始：解析有界本地图像、校验媒体类型、
一次性读取，并计算规范源证据。只有在此之后，字节才会交给注入的提供者。
提供者必须声明其源传输策略，且不能替换由 `OcrClient` 记录的源路径、
媒体类型、大小或 SHA-256。两个内置提供者都复用 A3S Power 的嵌入式、
模型无关推理基底；二者均不启用 Power 的 HTTP 服务，也不自行打开监听器。

## 职责边界

A3S OCR 识别一张有界图像并返回 OCR 证据。它不是文档解析器。

| A3S OCR 拥有 | 委托给 A3S Power | 本仓库之外 |
| --- | --- | --- |
| PP-OCRv6 与 Unlimited-OCR 的拓扑、资产、预处理、解码、标签、置信度与源像素几何 | 类型化设备、准入、权重完整性与驻留、取消、私有遥测、TEE 兼容控制与执行回执 | Office/PDF 页面清单、渲染、跨页层级、证据对账、智能体规划与文档检查点 |

PDF 栅格化与 Office 解析属于其各自拥有组件。文档级消费者（如 A3S Parser）
可在更大图中保留 `OcrResult` 块与回执，但不得将 OCR 模型所有权迁入解析器。
Power 保持模型无关，不包含 OCR 架构或资产。

## 快速开始

在已安装 A3S Use 的前提下，读取图像前先检查已配置的提供者：

~~~bash
a3s use ocr doctor --json
~~~

诊断会报告提供者、引擎、模型、就绪状态以及 `sendsSourceOffDevice` 策略。
对默认本地提供者，若诊断建议安装，请先安装固定版本的模型包，然后再抽取：

~~~bash
a3s install use/ocr
a3s use ocr extract ./scan.png --json
~~~

独立二进制暴露相同的领域操作：

~~~bash
a3s-use-ocr doctor --json
a3s-use-ocr extract ./scan.png --json
a3s-use-ocr serve --mcp
~~~

### 在 Rust 中嵌入客户端

默认 feature 集包含 PP-OCRv6、MCP 与 CLI：

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

当应用只需要中立契约与客户端时，使用 `default-features = false`。

## 分阶段批量抽取

`OcrClient::extract_batch` 接受稳定的调用方自有槽位 ID 与类型化阶段集合。
即使源校验或某个提供者阶段失败，它也始终按调用方顺序返回槽位：

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

提供者中立的阶段词表为 orientation、preprocessing、layout、text、table、
formula 与 seal。提供者描述符声明其可完成的子集；未实现的阶段返回
`unsupported`，绝不会从文本推断。面向既有提供者的兼容适配器仅支持
text 阶段。PP-OCRv6 目前声明 preprocessing 与 text，其中 preprocessing 表示
有界图像解码与规范化。它尚未声明表格或印章检测。单独构造的
`DocumentFastOcrProvider` 将该文本提供者与固定的 SLANet-Plus 有线表格模型
组合，并声明 preprocessing、text 与 table。当单独固定的 PicoDet layout 包
已配置时，同一显式提供者也声明印章检测。它保持可选接入，因为每新增模型
及其限制都必须对宿主可见。

分阶段批量 schema v2 要求每个已完成的 table 或 seal 阶段，在精确的源图像
像素画布上携带有界类型化载荷。表格证据保留检测到的表格区域、可选网格
维度、合并单元格跨度、文本，以及提供者实际给出的单元格几何。印章证据
保留其精确区域、可选识别结果、可见标记被裁切时的规范画布边，以及模型
是确认了该页上的对象还是仅保留了 `boundary-candidate`。客户端在发布结果前
会拒绝无效多边形包络、画布外区域、重叠或网格外单元格、伪造裁切、重复
身份与无界文本。
跨页表格与骑缝章对账仍由 Parser 负责；OCR 只产出页内证据，从不自行拼页。

### 可选接入的有线表格提供者

将 `A3S_OCR_SLANET_PLUS_MODEL_DIR` 设为经审阅的本地包，且清单必须精确为：

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

该提供者从相交页规则接纳保守的有线表格裁切，经 A3S Power 运行固定的
488 像素 SLANet-Plus 编码器，在本地解码自回归结构与单元格四边形，并按
源像素几何将 PP-OCRv6 文本块分配到模型单元格。仅有行候选绝不会作为表格
证据发布。无框线表格仍不受支持，且页内片段不会被 OCR 标为跨页延续。

### 可选接入的模型印章位置

将 `A3S_OCR_PICODET_LAYOUT_MODEL_DIR` 设为包含已转换 `model.safetensors` 的
经审阅本地目录。入库图是固定版 PaddleOCR `PicoDet-L_layout_3cls` 原始头的
确定性 lowering；生产既不加载 Paddle，也不加载 Python。模型拥有 `seal` 类别，
宿主在源像素坐标中执行有界分数过滤与 NMS。

常规页面路径使用一个 640 像素全页视图，加上固定的左右边缘条带。达到经审阅
阈值的全页检测为 `confirmed`。低置信度边缘证据绝不会被提升：它作为
`boundary-candidate` 返回，必须触及声明的源画布边，且在无下游对账时仍不可
作为已确认对象发布。

对已接纳序列，调用方可显式用 `with_adjacent_predecessor` 将槽位绑定到其
紧邻前驱。当前驱包含有界边缘候选时，OCR 最多为该边在当前页再跑一次本地
视图。这在保留的两页骑缝章夹具中恢复了狭窄的右边缘片段，同时第二页独立
保留其三枚内部印章。邻接声明仅授权额外的页内证据采集；跨页匹配、提升与
规范几何仍由 Parser 拥有。印章文本识别与通用印章准确率分数尚未实现。

一次请求包含 1 到 256 个唯一槽位，以及在既有每图 32 MiB 限制之外至多
256 MiB 的已校验输入字节。畸形的请求或提供者输出形状会使调用失败。源、
模型加载与阶段执行错误仍以 completed、partial、failed、skipped 或
unsupported 结果附着在精确槽位上。结果还携带规范提供者指纹、每槽位模型
指纹以及仅摘要的执行回执；原始源字节、张量值与本地路径不会放入调度证据。

## 结果契约：OCR 与出处

提供者拥有识别。`OcrClient` 拥有证据信封。

| 由 `OcrClient` 拥有 | 由提供者拥有 |
| --- | --- |
| 规范路径、检测到的媒体类型、字节大小、SHA-256 | 识别文本与模型身份 |
| 输入边界与受支持图像签名 | 可选置信度、类别、多边形与边界框 |
| 提供者输出校验 | 就绪消息与提供者特定警告 |
| 最终 `OcrResult` 组装 | 声明的离设备源策略 |

原生结果还可包含 `executionReceipts`。每条回执绑定模型族与修订、精确权重
摘要、Power 运行时/设备身份，以及规范输入/输出摘要。下游解析器应与 OCR
证据一并保留这些回执。

稳定结果形态将源与 OCR 证据放在一起：

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

类别、置信度与几何均为可选。`category.rawLabel` 保留有界提供者标签，而不
声明提供者分类法已封闭；`category.role` 是保守的提供者中立解释。组件框
保留精确提供者几何，而 `boundingBox` 是其兼容包络。OCR 输出是源自源的
证据，而非已验证的源文本。

## 提供者

提供者选择是类型化对象，绝不是原始后端名开关。

| 提供者 | OCR 自有实现 | 执行基底 | 源边界 |
| --- | --- | --- | --- |
| `PpOcrV6Provider` | 检测/识别图、图像流水线、DB/CTC 后处理 | 嵌入式 A3S Power | 始终在设备上 |
| `DocumentFastOcrProvider` | PP-OCRv6 文本、SLANet-Plus 有线表格结构，以及可选 PicoDet-L 印章位置与类型化边界候选 | 嵌入式 A3S Power | 始终在设备上 |
| `UnlimitedOcrProvider` | 视觉塔、投影器、解码器、分词器、生成与 grounding | 嵌入式 A3S Power | 始终在设备上 |
| 自定义 `OcrProvider` | 由实现定义 | 由实现定义 | 须在其描述符中声明 |

### 默认：PP-OCRv6

默认 A3S 集成使用：

- 提供者 ID：`pp-ocr-v6`
- 引擎：`a3s-power-native`
- 固定包：`PP-OCRv6_small`
- 传输策略：仅本地

其流水线是显式的：

~~~text
bounded decode → cross-image letterbox → batched detection → per-slot DB
               → identity-bound crop plans → stable width sort
               → cross-image crop batches → CTC decode → ordered evidence
~~~

OCR 自有发布打包固定检测与识别 SafeTensors 及其推理配置。安装会校验
归档长度与 SHA-256，只解压四个声明文件，并记录精确的 Power 权重摘要。
嵌入式 SLANet-Plus 与 PicoDet 图身份对仓库中 LF 规范化的 JSON blob 做哈希；
`.gitattributes` 与摘要测试会拒绝平台换行漂移。安装与修复保持显式：

~~~bash
a3s install use/ocr
a3s install use/ocr --force
~~~

模型下载约束连接建立与停滞读，但不施加总传输截止时间，因此健康的慢链
仍可完成固定归档。中断的正文从精确已校验字节范围重试；若服务器忽略范围，
则重启暂存文件而非追加。同一有界重试预算覆盖瞬时连接与源站失败。完整
归档在激活前仍必须匹配其固定长度与 SHA-256。

`A3S_OCR_MODEL_DIR` 可将开发构建指向显式模型包。`A3S_USE_OCR_HOME` 可覆盖
受管模型根，用于打包、测试或隔离安装。提供者通过 Power 的共享准入、设备、
限额、完整性、取消与回执机制执行经审阅的 OCR 自有图计划。它不需要
ONNX Runtime、Python、PaddlePaddle、子进程、推理服务或 Web 监听器。

分阶段 PP-OCRv6 批处理复用精确、惰性加载的 Power 模型会话，并根据实时
主机/设备内存快照规划确定性连续微批。每个已接纳微批在其槽位间持有一个
取消令牌、设备许可与引擎锁，并发出带会话声明、计划摘要、批索引/计数、
槽位数与队列证据的 schema-v4 回执。检测预处理与 DB 后处理最多使用 16 个
有界工作线程，并保持精确槽位顺序。快速检测器将最长边限制在 896 像素，
同时多边形映射回不可变源，识别则裁切该原始图像。对至少有 32 级通道变化
的源，空快速结果会触发一次最长边 4,000 像素的标量质量重试；两次检测回执
均会保留。该重试保护空结果质量，但不保证不会漏掉部分小字。

当每个槽位至少保留 90% 画布填充时，不同缩放尺寸的图像会在同一归一化黑色
画布的左上角做 letterbox。OCR 会确定性拆分更低填充的形状离群值，以及经审阅
峰值中间态将超过 Power 张量元素限制的任何队列。每个兼容队列最多含 16 张
图像，并执行一次动态 `[B,3,H,W]` 检测图调用。Power 校验主轴组装与输出分区；
OCR 保留每个槽位的内容范围，从 DB 后处理中排除填充，并将多边形经该范围
映射到源像素。随后 OCR 在已接纳图像间展平检测到的裁切，同时保留精确槽位、
检测与阅读顺序身份。它将动态识别宽度稳定排序为至多八个裁切的规范组，
且最宽画布不超过最窄画布 16 像素。由于每个识别画布至少 320 像素宽，经审阅
边界最多增加 5% 右侧填充，同时折叠像素级裁切抖动；更大宽度差仍保持分离。
相邻规范组仅在最终画布宽度已完全相同时合并，硬物理上限为 32 个裁切。这既
不改变填充，也不改变模型输入值。透视裁切与识别张量使用共享 Rayon 工作池，
并恢复相同确定性顺序；标量相对批量的张量测试是字节精确的。规划器只物化
活动组，并将块与回执恢复到其源槽位。无界宽度混合仍被禁止，因为
PP-OCRv6 识别具有全局宽度上下文，可能改变解码文本。失败的共享图调用会经
标量路径重试其受影响裁切，使非取消失败保持隔离；取消仍会终止已接纳请求。
仅含空白的识别结果会从公开块中省略，而不是发布无效空证据；该过滤在推理
之后运行，不是检测器置信度捷径。

有界宽度发布门使用 SHA 固定的 Parser 栅格。三页跨页表格夹具保留其精确文本
指纹、`6x6/29`、`8x7/25` 与 `3x6/17` 网格/单元格，以及两条延续边。两页骑缝章
夹具保留其精确文本指纹、三枚完整印章、两个经 IoU 校验的右边缘片段，以及一个
延续身份。全文档门将其扩展到全部六页表格与全部 29 页骑缝章：CUDA 保留精确
文本与结构化几何指纹、71 个表格单元格、两条表格延续、12 枚完整印章，以及两个
已对账边界片段且无未解决候选。在指定的开发用 RTX 4090 上，等画布合并加并行
识别预处理首先将 29 页 CUDA 中位从 8.400 秒降至 6.834 秒。随后当前字节精确
GELU 融合将五次运行、交替顺序的中位从六页表格文档的 1.489 秒降至 1.463 秒
（4.101 页/秒），以及 29 页骑缝章文档的 6.255 秒降至 5.838 秒（4.968 页/秒）。
后续通道偏置融合 A/B 在当前机器负载下对表格门用九次交替运行、对印章门用
五次：中位分别从 1.387 降至 1.340 秒（4.326 至 4.478 页/秒，延迟低 3.4%），以及
从 6.067 降至 5.960 秒（4.780 至 4.866 页/秒，延迟低 1.8%）。每次运行都保留相同
文本、表格延续、单元格、印章位置与边界片段断言。随后的 LayerNorm-affine-tail
A/B 使用相同交替协议：九次表格中位从 1.270 降至 1.215 秒（4.724 至 4.938 页/秒，
延迟低 4.3%），而五次印章中位基本持平，为 5.848 对比 5.840 秒（4.959 对比
4.966 页/秒）。当前单次 CPU 捕获分别耗时 46.334 秒（0.129 页/秒）与 334.596 秒
（0.087 页/秒）。CUDA 结果仍低于完整精细解析的 10 页/秒目标。这些是夹具特定的
正确性与延迟诊断，而非语料范围的 OCR 准确率或吞吐声明。

识别不再在主机上物化完整的 18,710 类概率行。OCR 在 Power 执行设备上应用
确定性的模型自有投影，并为每个 CTC 时间步传输
`[class index, score, source-finite marker]`。反向轴归约保留标量解码器的
末类平局规则，同时标记覆盖每个源概率而非仅所选分数。对经审阅的
`[1,40,18710]` 输出，这将主机物化与回执哈希从 2,993,600 字节降至 480 字节
（**6,236.7×**）。投影修订是模型/会话执行身份的一部分，执行回执提交给 CTC
解码所消费的精确投影张量。

固定 Power 运行时还融合了经审阅的 CUDA 乘数为一的深度可分离卷积：17 个检测
层与 14 个识别层现在每个节点执行一个 F32 内核，而不是每个内核位置一次设备级
乘/加序列。检测偏置在最终项之后于同一内核中应用。显式四舍五入到最近算术
保留先前累加顺序，且 Power 的所选设备奇偶校验门是字节精确的；CPU 与不受支持
的张量布局保留既有路径。OCR 仍拥有图清单与端到端输出奇偶校验。

同一固定 Power 修订在 CUDA 上私有融合相邻、单消费者的 F32
`HardSigmoid`-到-`Mul` 通道门。经审阅 OCR 计划含 13 个此类检测位点与五个识别
位点；图清单测试锁定这些计数。每个匹配的 `[N, C, 1, 1]` 门作用于
`[N, C, H, W]`，用一个字节精确内核替换原先四次激活传递加广播乘法，每位点
去掉四次启动与四个中间缓冲。图拓扑、回执与 OCR 所有权不变。CPU 以及每个
未经审阅的 dtype、形状、广播形式或布局仍保留逐节点执行。

识别还包含 13 条相邻、单消费者的分解 GELU 链，每条表示为带三个标量初始化器
的 `Div`-`Erf`-`Add`-`Mul`-`Mul`。固定 Power 执行器在模型加载时捕获这些标量
一次，并将每条链作为带显式除法、加法与乘法舍入边界的一个 CUDA 内核运行。
其字节精确内核与完整图门在每条链上去掉四次启动，而不重写 OCR 自有图；因此
保留的 29 页印章门中的 138 次识别调用避免了 7,176 次启动。CPU 以及每个未匹配
图、dtype、布局、设备、输出或共享中间态仍保留普通逐节点执行。

识别图另外包含 10 个 `Conv`-偏置-ReLU 前缀、13 个喂入这些 GELU 链的
`Conv`-偏置前缀，以及五个喂入门控 HardSigmoid 乘法的 `Conv`-偏置前缀。OCR
自有拓扑测试锁定这 28 个计数、相对卷积输出通道的精确 F32 `[1,C,1,1]` 偏置
形状、身份深度与私有消费者关系。固定 Power 执行器将每个卷积保留在既有后端，
并将通道加法折叠进随后的字节精确 CUDA 激活。每次识别调用再避免 28 次完整
张量启动与缓冲；保留的 138 次印章门调用避免 3,864 次。测得的快速路径要求
有启动界的 32 位通道索引。CPU 以及每个未经审阅的偏置、形状、拓扑、设备、
dtype 或布局保留普通图。

识别中的五个分解末轴 LayerNorm 块保留其两次均值归约、中心化与平方，同时
Power 私有融合每个精确的 `Add(epsilon)`-`Sqrt`-`Div`-`Mul(scale)`-`Add(bias)`
尾部。OCR 自有清单测试锁定五个相邻私有窗口、标量 epsilon 与 120 元
scale/bias 初始化器。显式 F32 舍入边界使 CUDA 尾部与原先五个节点字节精确。
每次识别调用再避免 20 次启动与中间缓冲；保留的 138 次印章门调用避免 2,760 次。
CPU 以及每个未经审阅的拓扑、形状、设备、dtype 或布局保留普通执行。

当前质量证据覆盖固定的 30 块官方图像，以及在 144 DPI 渲染的清晰 8 点与
12 点 PDF 文本。五点合成文本未通过精确发布，不是受支持的质量声明。指定
硬件的 Parser 集成门由消费方 Parser 记录；本 crate 不会将该工作负载变为通用
OCR 吞吐声明。

Linux CI 安装该精确固定包，并在 CPU 上执行两个经审阅图。该门检查规范
Power 权重摘要、精确输出形状、条目计数，以及零张量检测与识别夹具的字节长度。
随后下载 PaddleOCR 经 SHA-256 固定的 `general_ocr_002` 图像，并执行完整 Rust
流水线：缩放、检测、DB 后处理、阅读顺序排序、透视裁切、批量识别、CTC 解码、
源坐标多边形，以及八条 Power 执行回执。30 个输出块对照用 Paddle 3.3.1 与
PaddleOCR 3.7.0 生成的参考进行检查，使用显式文本、分数与四点坐标容差。同一
门比较一个官方裁切在标量与跨图批宽度二下的结果，要求文本与几何相同、识别
置信度在 `0.00001` 内、一条共享识别回执，以及精确 2× 输入张量大小。Paddle、
Python 与 ONNX Runtime 不是本 crate 的测试或运行时依赖。

`a3s-use-ocr-execution-bench` 为该固定图像增加严格、无路径的真实提供者基准。
它将首次惰性模型会话与暖执行分离，每毫秒采样进程 RSS，保留检测与识别的
Power 指纹，并拒绝输出漂移。固定对象在上游名为 `.png`，但具有 JPEG 字节
签名；源证据跟随字节。调试或已修改树的报告仅作诊断。发布流程与声明边界见
[PP-OCRv6 Execution Baseline Protocol](docs/execution-baseline.md)。

Power/OCR 所有权边界、模型转换与安装完整性、执行回执以及 TEE/隐私发布门见
[Native Inference Architecture](docs/native-inference.md)。对齐的 Power/OCR/Parser
交付序列与 TurboOCR 衍生工作流见 [`ROADMAP.md`](ROADMAP.md)。

### 可选：baidu/Unlimited-OCR

启用 `unlimited-ocr` feature 可在进程内运行经审阅的 3B 视觉-语言模型。
A3S OCR 拥有原生 Rust 模型拓扑、分词器、预处理、生成循环、修订固定与
grounding 解析器。A3S Power 提供共享设备、准入、权重完整性、驻留、路由、
取消、遥测与回执机制。

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

`UnlimitedOcrConfig::from_env` 从 `A3S_UNLIMITED_OCR_MODEL_DIR` 读取相同本地路径。
提供者创建是惰性的：不执行模型下载、进程启动、网络请求或套接字绑定。会话
加载仅接受固定上游修订
`07dea832e22aefee32ad281d4b80551282e1c168`，校验精确分词器与处理器资产，并
请求 Power 执行单一完整 SafeTensors 哈希与清单校验路径，包括任何显式配置的
已验证副本。经审阅的主权重文件恰好为 6,672,547,120 字节，SHA-256 为
`2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6`。
不可跳过的官方清单门仅解析修订
`07dea832e22aefee32ad281d4b80551282e1c168`，校验 Hugging Face 仓库提交以及链接
文件大小与 SHA-256，并范围读取 334,632 字节的 SafeTensors JSON 头，而不是下载
6.7 GiB 载荷。它检查固定小资产摘要、官方索引、全部 2,710 个 BF16 张量名、
形状与字节范围、精确 6,672,212,480 字节张量载荷布局，以及 OCR 自有规范清单
摘要。会话加载在推理前将 Power 的完整哈希清单与该同一摘要比较。此门证明
检查点身份与拓扑。单独的本地数值门执行完整官方检查点，并使模型输出接受
独立于清单接受。

数值门下载既有 SHA-256 固定的 PaddleOCR 登机牌图像，在 Rust 中导出固定
640×528 无损裁切，并通过生产生成所用的同一 KV-cache、no-repeat 与解码器循环
对全部 64 个上游 CPU 参考 token 打分。它记录每个期望 token 的秩与 logit 增量，
然后执行第二次自由运行的贪心解码。带 Apple Accelerate 的 CPU 精确匹配全部
64 个参考 token。Metal 精确保留前 15 个，且至多有两个秩-2 边界，最大 logit
增量为 0.25；可见差异是一个可选前导标点与三像素标题框边。两条路径都必须
返回相同的三个 `header`、`title` 与 `text` 块、经审阅文本，以及该三像素界内的
源像素几何。当审计预期提供完整教师强制 token 相等的后端时，设置
`A3S_UNLIMITED_OCR_REQUIRE_EXACT_PARITY=1`。

原生前向路径遵循权威上游实现：

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

一次逻辑抽取在完整视觉、投影器、解码器与 grounding 流程中持有一个 Power
许可与取消令牌。路由专家使用 Power 的精确批并集与默认私有的路由遥测，而
不是第二个 OCR 本地缓存。缓存驻留默认仍为零。类型化、可选接入的
`ResidencyBudgetPolicy` 请求所选 Power 运行时发现有界主机/CUDA/Metal 容量，
并从显式分数、预留、上限与运行时限制推导缓存字节；Metal 统一内存只计一次。
手动缓存字节与自动预算互斥。容量快照既不持久化，也不加入遥测或执行回执。
在任一显式缓存模式下，有界专家预取与共享专家计算重叠，并使用 Power 的
LFRU/LRU 放置。丢弃等待中的识别 future 会取消该共享令牌；阻塞原生工作线程
随后在其有界预处理、视觉与解码器取消点停止。
提供者发出一条最终回执，绑定源图像摘要、经审阅权重集合、Power 设备与用户
可见 UTF-8 文本。

CPU 随 `unlimited-ocr` 可用；`unlimited-ocr-accelerate` 启用 Apple Accelerate
CPU 内核，同时保留 BF16 模型边界。使用 `unlimited-ocr-metal` 构建以获得显式
Apple Metal 设备，或使用 `unlimited-ocr-cuda` 获得显式 NVIDIA CUDA 设备。当
请求的加速器不可用时，类型化设备选择失败关闭。在 Power 的 TEE 部署中运行时
保留模型完整性、资源边界、私有遥测与回执保证；源字节与详细路由数据绝不会
被此提供者导出。

提供者在原生生成循环中应用上游单图提示与 no-repeat n-gram 策略，并保留
生成的 Markdown。它严格解析上游模型实现中审阅过的两种 grounding 形式：

~~~text
<|ref|>title<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>text
<|det|>text [x1, y1, x2, y2]<|/det|>text
~~~

Unlimited-OCR 坐标使用上游后处理器文档中的闭区间 `0..=999` 基
（[upstream postprocessor](https://huggingface.co/baidu/Unlimited-OCR/blob/07dea832e22aefee32ad281d4b80551282e1c168/modeling_unlimitedocr.py#L62-L111)）。
A3S OCR 解析已验证输入尺寸，并将有效非图像 grounding 映射为类型化源像素
`OcrBlock` 证据。每个有效组件框按模型顺序保留，且 `boundingBox` 仍是兼容用的
有界并集。有界原始标签与保守角色一并保留：显式标题、heading、段落、表格、
说明、公式、页眉/页脚、脚注、页码与代码获得匹配角色；其他有效标签保持
`unknown` 而非被提升。上游分类法被有意视为开放。

实现不会将任何模型文本当作代码求值，不伪造置信度，也不为缺失、畸形、越界、
空、仅图像或经 EXIF 变换的 grounding 发出几何。它从不信任生成的图像路径。这
遵循上游加载器的 EXIF 转置行为，而不会将变换后坐标误标为未变换源像素。降级
grounding 通过一条有界警告保持可见，同时保留生成文本。诊断校验本地资产清单，
而不会将 6.7 GiB 哈希做两次；首次会话打开完成 Power 强制的完整检查点校验。

### 提供者接口

`OcrProvider` 保持对象安全、`Send + Sync`，并独立于具体提供者依赖：

~~~rust
#[async_trait::async_trait]
pub trait OcrProvider: Send + Sync {
    fn descriptor(&self) -> OcrProviderDescriptor;
    fn diagnostic(&self) -> OcrProviderStatus;
    async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput>;
}
~~~

用 `OcrClient::with_provider(provider)` 或
`OcrClient::from_provider(Arc<dyn OcrProvider>)` 注入实现。描述符必须包含稳定
提供者 ID、引擎名与离设备源策略。

## CLI 与 MCP 界面

| 界面 | 入口点 | 提供者行为 |
| --- | --- | --- |
| A3S Use | `a3s use ocr ...` | 保留的内置路由；PP-OCRv6 为当前默认 |
| 独立 CLI | `a3s-use-ocr ...` | 等价的 `doctor`、`extract` 与 `serve --mcp` 操作 |
| Rust 库 | `OcrClient` | 接受任何类型化提供者 |
| 标准 MCP | `OcrMcpServer::new(client)` | 暴露 `ocr_doctor` 与 `ocr_extract` |

`OcrMcpServer` 将提供者的源传输策略投影到 `ocr_extract` 工具注解中。因此
自定义离设备提供者对 MCP 宿主保持可见，而不会看起来像仅本地读取。

## Feature 标志

| Feature | 增加内容 |
| --- | --- |
| `power-runtime` | 模型无关的嵌入式 A3S Power 运行时；从不启用其 server feature |
| `ppocr-v6` | 本地 PP-OCRv6 提供者、原生图计划、安装器、图像流水线 |
| `ppocr-v6-cuda` | 经 Power 的 NVIDIA CUDA 路径进行 PP-OCRv6 与 document-fast 表格/印章推理 |
| `benchmark` | PP-OCRv6 真实图像冷/暖执行基线二进制 |
| `unlimited-ocr` | 原生 CPU Unlimited-OCR 模型、分词器、图像流水线、生成与 grounding |
| `unlimited-ocr-accelerate` | Unlimited-OCR 加带经审阅 BF16 操作边界的 Apple Accelerate CPU 内核 |
| `unlimited-ocr-metal` | Unlimited-OCR 加 Power/Candle Apple Metal 设备路径 |
| `unlimited-ocr-cuda` | Unlimited-OCR 加 Power/Candle NVIDIA CUDA 设备路径 |
| `mcp` | 提供者中立的标准 MCP 宿主 |
| `cli` | 独立 CLI；组装 PP-OCRv6 与 MCP |
| default | `ppocr-v6`、`mcp` 与 `cli` |

## 输入与信任边界

- 输入是 1 字节到 32 MiB 之间的常规本地文件。
- 受支持签名为 PNG、JPEG、WebP、GIF、BMP 与 TIFF。
- URL 与 PDF 栅格化不在当前客户端契约内。
- 提供者不能替换由 `OcrClient` 创建的规范源证据。
- 页码从 1 开始；返回的置信度值必须有限且位于 0 到 1 之间。
- 提供者标签与组件框列表有界；组件列表必须与其兼容包络精确一致。
- 两个内置提供者从不将源字节传出设备。
- Unlimited-OCR 源框与类别仅从有效、有界的 `0..=999` grounding 与已解码源图像
  尺寸发出；畸形标记绝不会变成框或语义声明。
- 模型安装、修复与检查点获取绝不会隐藏在抽取内部。
- 内置嵌入式推理边界不包含 ONNX Runtime、外部 OCR 服务、HTTP 客户端/服务器、
  浏览器自动化、Python 运行时、子进程推理或网络监听器。

## 开发

从本 crate 仓库运行检查，而不是从 A3S 单体仓库根目录：

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

该库依赖已发布的 `a3s-use-core` 机器契约。A3S Use 在组装内置路由、打包
Skill 与模型资产时固定不可变 OCR 修订。

分阶段 PP-OCRv6 集成固定可发布的 A3S Power 0.8.0 修订
`2939668e8ad38e4d3f564144d01c2a5020aa39de`。源码构建与 CI 执行该精确 Git 修订。
包校验还会解析声明的 `=0.8.0` registry 依赖，因此在同一 Power 发布在
crates.io 可见之前，包门保持关闭。本仓库不应出现 path 或
`[patch.crates-io]` 覆盖。

<details>
<summary>发布所有权</summary>

本仓库拥有提供者接口、默认 PP-OCRv6 实现、原生 Unlimited-OCR 实现、测试、
模型出处、Skill 内容、crate 发布与平台归档。A3S Use 拥有内置路由、所选默认
提供者、能力投影、组件策略与最终产品组装。发布通过不可变修订与 SHA-256
绑定制品对接。

</details>

## 许可证

按 [MIT License](LICENSE) 许可。模型与运行时出处见
[Third-Party Notices](THIRD_PARTY_NOTICES.md)。
