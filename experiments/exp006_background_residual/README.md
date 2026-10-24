# 实验6：冻结 Pyronear，加入正常背景预测与局部残差分支

状态：2026-10-22，baseline/seed_42完整特征提取、训练、阈值校准和正式独立评价均已完成。[深入分析](analysis_20261022/RESULTS.md)已独立核算793条测试候选，并用冻结权重复算融合分数。五种方法均检出8/10；学习残差中位延迟2.98分钟，原版4.49分钟，但正常9.974摄像头小时误报由0增至5段，尚未证明综合优势。高分辨率1536及43/44种子尚无正式结果。

## 目的与当前定位

第五次实验发现：正确烟羽位置不保证早期薄烟获得高分；部分背景持续高分；校准还可能放大长期背景优势。本轮验证一个假设：显式预测正常背景，并将实际特征与预测特征的差异用于复核，是否能补充现有外观和时序证据。

这是对 [EARL 正常场景预测与残差机制](https://www.mdpi.com/1424-8220/26/19/6020)的借鉴和适配，不是完整 EARL 复现，也不是已成立的技术创新。背景变化不自动等于烟，低残差也不直接否决烟。本轮不照搬其延迟损失。

**整套系统继续使用官方训练好的 Pyronear。没有再复制一套未训练的 Pyronear，也不使用20个事件从头训练原模型。**

- 官方 YOLO、关联、裁剪、DINOv2 和时序分类器保持冻结。
- 新增预测器共用官方 DINOv2 的 patch 特征，只训练预测器和小型融合头。
- 所有组在同一检测设置内使用相同候选和轨迹；人工框只用于标注和评估，不参与自动推理定位。
- 首先运行 0.10/1024，再在 0.10/1536 下复核。关联与裁剪参数保持官方原值。

## 数据与划分

数据根目录：`D:\AIC\AIC-seek\data\exp006_background_residual`。

| 划分 | FIgLib新火情 | 火情图片 | HPWREN正常候选片段 | 正常候选图片 |
| --- | ---: | ---: | ---: | ---: |
| train | 20 | 1610 | 10段，两小时/段 | 1200 |
| val | 10 | 810 | 5段，两小时/段 | 599 |
| test | 10 | 788 | 5段，两小时/段 | 600 |
| 合计 | 40 | 3208 | 20段，计划40摄像头小时 | 2399 |

共5607张活动原图，下载无失败；SOURCE.md记录总字节数，manifest.json记录每张原图的哈希和尺寸。实际有效观测时长按时间戳计算，不将缺帧或未确认的画面当作正常观测。初次下载后识别出的5个夜间序列已另存excluded_night，不参加本轮实验。

- FIgLib：[官方索引](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html)。每个事件取一个固定彩色机位；按不同日期保守去重，并避开同年重复的命名事件。实际偏移存在秒级抖动和缺帧，不强制改名或补帧。
- HPWREN：[官方相机说明](https://www.hpwren.ucsd.edu/cameras/help.html)。从官方清单获取原始 JPEG，分散到2026-10-03、08、12、16、19及上午、正午、下午时段。Q4/Q5/Q6是加州本地时间分块，不是UTC分块。
- 以来源**站点前缀**隔离三组，比仅隔离摄像头视角更严格；当前train/val/test含正常片段分别覆盖18/8/8个站点。正常候选覆盖20个站点，也遵循相同划分。
- 排除前面开发使用的 sp、lp、hp、syp 站点和 Alison、Cranston、West、Whittier。旧事件仅用于程序兼容性检查。
- 所有下载片段仍需人工确认；“正常候选”不等于已经标注的无烟数据。FIgLib官方0分钟是首次可见烟羽参照，不是真实点火时刻。
- 与官方模型原训练集的重叠尚未核实，只能称“未参与本轮训练和调参的测试事件”。

40个事件/40小时是冻结特征原型的预算起点，不代表20个训练事件必然足够，更不保证超过官方模型。关注事件与站点覆盖，后续依据独立验证表现和训练规模曲线扩充。10小时独立正常候选只够初筛长期误报。

## 程序与输出

| 文件 | 用途 |
| --- | --- |
| protocol.json | 固定输入、模型哈希、网络大小、训练和告警参数 |
| data_selection.json | 固定各序列来源、分组、时间戳和原图URL |
| scripts/download_data.py | 官方索引解析、选样、原图下载及断点续传 |
| scripts/features.py | 冻结官方模型，缓存原分数和因果局部特征 |
| scripts/learning.py | 正常预测器、对照分支和融合头训练 |
| scripts/evaluation.py | 验证集定阈、冻结阈值后的独立测试 |
| scripts/review.py | 共享标注、本地复核页面、VIA导入 |
| scripts/review.html | 连续候选缩略图、播放、整组/整序列确认和保存提示 |
| scripts/review_assist.py | 仅供复核的空间分组、相似特征预标注、批量保留已有标签和可靠写盘 |
| scripts/run_experiment.ps1 | PowerShell统一入口 |
| tests/test_exp006.py | 因果性、标注语义、分组泄漏与指标测试 |
| tests/test_review.py | 批量保存、已有标签保留、不确定优先、HTTP保存与备份测试 |

原图、官方索引快照、下载清单和共享标注留在数据根目录。运行产物位于 `outputs/<RunId>/<setting>/`：检测缓存、候选记录、特征缓存；每个随机种子的新增权重、验证阈值和独立测试结果写入 `seed_<Seed>/`。官方模型压缩包不修改。

## 结构与对照

一次DINOv2编码保留patch特征，汇聚成4×4局部网格。当前候选的固定裁剪矩形应用于当前及过去图像，避免框抖动制造变化。预测器用过去5帧预测下一帧；融合使用最近3次预测残差，共需8张不同时间的真实图片。间隔过大或历史不足时，不伪造输入，所有新增方法退回同一官方概率。

正常预测器：共享的单层GRU，隐藏维度128；仅用人工确认的train正常片段训练，用val正常片段选停止轮次。正常训练在每帧3×3固定区域采样，每个区域再保留4×4局部特征。编码器冻结，因此不需要EMA编码器防止表征塌缩。

融合头：原始烟雾logit、轨迹跨度、平均检测置信度、候选数，加对应分支证据；使用人工确认的候选烟/背景标签训练。预测器进入融合训练后也冻结，不在线吸收测试画面。

| 方法 | 改动与目的 |
| --- | --- |
| original | 原模型；同时保留官方默认阈值参考和验证集匹配工作点 |
| calibration_only | 只用原L/T/C/N拟合线性校准，分离校准收益 |
| history | 同样的新增历史画面，普通GRU融合；检查是否只是更多历史有用 |
| mean_residual | 用过去5帧特征均值作正常参考；检查简单差异是否足够 |
| learned_residual | 学习预测正常特征，再融合残差；检验学习预测的额外价值 |

最新方案先冻结全部原模型，不默认重训原时序头。四个可训练对照使用同一候选标签和划分。学习预测器额外使用的自监督任务、参数量和训练窗口数量均记录在检查点；不能将全部提升直接称为架构创新。

默认AdamW，学习率0.001，batch64，最多30轮，val连续5轮不改善则停止。原型参数均在protocol.json中；改参数后使用新的RunId。最终候选建议以42/43/44三个种子检查训练波动。

## 标注要求

`labels.json` 是所有实验组共用的唯一标注状态文件：

- 逐帧状态：visible / absent / uncertain；未标注保持未知。absent表示人工明确确认该帧整图无烟，不是“没画框”。
- smoke_boxes是归一化xyxy粗框；同一图片只画一次，各组和两种分辨率共用。
- 正常片段只有整段复核后才标为confirmed_normal；否则预测器不会将其当作正常训练数据。
- 若正常候选实际有烟、镜头移动或不可用，可在页面填写原因并排除；程序报告剩余有效观察量。不能因为模型对云雾误报就排除困难正常背景。
- 候选烟/背景身份由人工复核；真实烟还须确认是否属于目标事件。候选身份随检测设置区分，避免1024和1536的不同框误用同一个标签。
- 已有VIA空框不自动转成absent：早期不可辨和后期省略框仍可保留未知。有框可导入为visible；旧开发事件不会自动进入新训练集。
- 人工框不替换检测框。无法判断的候选保留未知，不强行记为正或负。

## 分组复核，减少重复点击

已保存的人工标签在 `data/exp006_background_residual/labels.json`，关闭网页或在终端按 Ctrl+C 都不会清除。界面显示“已保存到磁盘”后即可退出。新画的框仍需点击相应帧状态才能保存，不能仅画框后离开。

完成 Extract 后可执行 Prelabel，再用相同 RunId 和 Setting 启动 Review：

```powershell
& $Runner -Stage Prelabel -RunId $RunId -Setting baseline -ReviewSplit develop
& $Runner -Stage Review -RunId $RunId -Setting baseline -ReviewSplit develop
```

1. 选择 FIgLib 火情片段，模式选择“分组复核（推荐）”。同一位置连续候选按每组最多12个整理，全部显示缩略图，可播放本组，也可点击缩略图看整图。
2. 预标正确时点击“本组预标注无误，确认保存”；需要纠正时用“本组是烟／背景／不确定”。原人工标签默认保留，只有主动勾选“同时修改已有人工标签”才批量修改。
3. 也可只浏览所有分组，纠正有问题的组，最后点击“本序列预标注已看完，确认并保存”，一次接收剩余预标注。按钮在各待复核组的缩略图被浏览后启用；不确定仍保存为不确定，不参与正负样本训练。
4. “人工首次可见”是可选辅助时间参照，每场至多设置一次；未填写时仍可按官方起点评价，人工参照延迟留空。不会将此前图像自动判为背景，也不会用模型首次检测时刻冒充人工首次可见。HPWREN正常片段只需整段确认，不要求逐张点击。

分组只为方便审阅，不改实验的轨迹关联。自动建议单独保存在 `outputs/<RunId>/<setting>/review_suggestions.json`，未经用户点击确认不会进入正式标签，也不能直接用于训练或测试。特征相似检索只参考已经人工确认的 train 样本，未采用原模型烟雾概率来制造真实标签。

本次另查看了剩余18个训练火情片段的各组首、中、尾画面，助手视觉预判及组内传播存于 `review_visual_hints.json`，对应图像和候选清单位于 `review_contact_sheets/`。这不是逐帧人工真值，中间帧和薄烟边界必须在页面复核。当前866个待确认候选整理为98组，预判烟435、背景135、不确定296，均未自动写入 labels.json。相似匹配证据不足的候选不会强行给出烟/背景结论。

每次服务首次修改标签前保留一份 `labels.before_session_*.json`，完整写盘后原子替换 labels.json；多页面版本冲突会提示刷新，避免覆盖另一页的新记录。开始本次修改前另备份了 `labels.before_bulk_review_20261022_125948.json`。正式训练仍只读取 labels.json。

## 启动命令

使用现有Python环境，无需另建环境。以下命令在同一PowerShell终端保留变量；Review运行时占用该终端，关闭页面后在终端按Ctrl+C退出。

```powershell
Set-Location D:\AIC\AIC-seek
$Runner = 'D:\AIC\AIC-seek\experiments\exp006_background_residual\scripts\run_experiment.ps1'
$RunId = 'exp006_main_20261021'

& $Runner -Stage Check
& $Runner -Stage Tests
```

下载中断时仅运行下面一条续传命令；已下载原图会校验复用。不要重新执行选样改变清单。

```powershell
& $Runner -Stage Data -Workers 8
```

先初始化共享标注；正常片段整段确认、火情首次可见可在这里记录。无需先给全部原图画框再提取；也可先执行后面的 Extract，再集中分组复核。页面不显示各改进模型的分数。

```powershell
& $Runner -Stage Prepare
& $Runner -Stage Review -RunId $RunId -ReviewSplit develop
```

浏览器打开 `http://127.0.0.1:8766`。可用方向键逐帧查看、鼠标画粗框；V=可见、A=人工确认整图无烟、U=不确定。正常片段浏览完整后点击“整段已确认无烟”。也可导入新事件的VIA文件：

```powershell
& $Runner -Stage ImportVia -Via 'D:\AIC\your_new_event_via.json'
```

随后提取完整候选与特征；这一步可以在标注未完成时执行，不训练模型。它会读取三组原图，但训练和选阈代码只使用train/val。

```powershell
& $Runner -Stage Extract -RunId $RunId -Setting baseline -Device cuda
```

生成预标注后重新打开页面，优先使用“分组复核”；需要逐个纠正时切换“单个候选详情”：S=烟，B=背景，U=不确定；确认是否属于目标火情。先完成train/val复核，未经确认的预标注不用于训练。训练需要两类人工确认的候选；空框不会自动补成负样本。

```powershell
& $Runner -Stage Prelabel -RunId $RunId -Setting baseline -ReviewSplit develop
& $Runner -Stage Review -RunId $RunId -Setting baseline -ReviewSplit develop

# 正常预测器 + 四个对照融合头；只训练新增参数
& $Runner -Stage Train -RunId $RunId -Setting baseline -Seed 42 -Device cuda

# 仅用验证集正常序列选择工作点，然后冻结
& $Runner -Stage Calibrate -RunId $RunId -Setting baseline -Seed 42 -Device cuda
```

固定方案后再复核test：确认正常片段和候选身份，按需要记录人工首次可见时刻，并确认本序列候选身份复核完成。人工首次可见不是正式评价的必填项；候选全为背景或没有候选的火情仍保留在事件分母中，不要求人为补一个正样本。测试标注不改变train/val指纹，不需要重新拟合阈值。

```powershell
& $Runner -Stage Prelabel -RunId $RunId -Setting baseline -ReviewSplit test
& $Runner -Stage Review -RunId $RunId -Setting baseline -ReviewSplit test
& $Runner -Stage Evaluate -RunId $RunId -Setting baseline -Seed 42 -Device cuda
```

独立结果位于：
`D:\AIC\AIC-seek\experiments\exp006_background_residual\outputs\exp006_main_20261021\baseline\seed_42\RESULTS.md`。

高分辨率另用独立设置，不覆盖原版：

```powershell
& $Runner -Stage Extract -RunId $RunId -Setting high_resolution -Device cuda
& $Runner -Stage Prelabel -RunId $RunId -Setting high_resolution -ReviewSplit develop
& $Runner -Stage Review -RunId $RunId -Setting high_resolution -ReviewSplit develop
& $Runner -Stage Train -RunId $RunId -Setting high_resolution -Seed 42 -Device cuda
& $Runner -Stage Calibrate -RunId $RunId -Setting high_resolution -Seed 42 -Device cuda
& $Runner -Stage Prelabel -RunId $RunId -Setting high_resolution -ReviewSplit test
& $Runner -Stage Review -RunId $RunId -Setting high_resolution -ReviewSplit test
& $Runner -Stage Evaluate -RunId $RunId -Setting high_resolution -Seed 42 -Device cuda
```

再次训练43/44种子时复用同一特征和标签，Train/Calibrate/Evaluate分别传入对应Seed，不重新选数据。

## 评价与继续条件

默认示例工作点：验证集误报不超过0.1次/摄像头小时，超阈画面时长不超过1%。它是原型配置，不是正式部署指标。各模型在验证集分别选阈，测试集不调阈；测试实际误报不一定恰好相同，应一起报告。

- 每个时刻以最高分候选作为报警区域，只有获胜区域经人工确认属于目标烟羽才计首次发现；目标低于背景时，即使目标也跨阈，仍不计正确定位报警。主延迟相对官方首次可见，另报人工首次可见参照。
- 同时报告事件命中、观察窗口内漏检、5/10分钟召回、检出事件条件中位延迟，以及各事件的完整观察时长。
- 正常序列按实际有效时间计算误报；同一整图连续阳性按5分钟间隔规则合并为一段。另报超阈时长占比，防止长期背景被合并成一次后看起来很好。
- 当前误报段是相机整图告警，不声称已完成不同区域的空间告警聚类。
- 没有候选的帧仍计入正常观测分母；没有检测到目标的事件仍计入事件分母。
- 新方法必须优于仅校准、普通历史和简单差异对照，且不以更多漏检换取更快的条件延迟，才值得继续投入。

训练标注不足、正常片段未确认、test身份未复核、特征提取仅为部分试跑时，程序会给出对应缺项并停止相关正式阶段，不生成虚假的正式结果。

## 本轮已执行的验证

`exp006_smoke_20261021`使用旧开发事件Alison的8帧检查GPU兼容性：官方模型冻结，成功提取8×16×384特征并完成原检测/时序推理；同目录还包含新train事件Norte前48帧的部分流程检查。`exp006_smoke_mixed_20261021`检查了Palisades前48帧及一个正常候选前48帧，正常特征缓存为48×9×16×384。这些部分回放没有保留候选轨迹，不提供识别效果结论；旧Alison兼容性检查保留了1条轨迹。

上述试跑均未进行正式训练，不计入新数据成绩。8项测试另覆盖了新增网络的最小训练循环与冻结性，使用合成张量而非正式标签。后续正式训练和独立测试已完成，成绩与局限见[深入分析](analysis_20261022/RESULTS.md)。

2026-10-22更新：baseline正式完整特征缓存已提取，训练/验证人工标注已开始。分组复核升级后共15项测试通过，新增覆盖批量保留已有标签、显式不确定优先、预标注不改变真值，以及隔离临时数据上的HTTP保存、版本冲突和备份。未在真实标注上自动执行“人工确认”。

后续进度：baseline/seed_42训练和阈值校准已完成。用户声明测试复核完成后，检查到592个火情候选均已有标签（烟404、背景188），5段正常片段均确认无烟，有效观测约9.974摄像头小时。依据该完成声明补齐了10个测试火情的复核完成标记，未改变候选身份、目标火情归属和人工时间；原文件另存为labels.before_test_completion_20261022_200223_984770.json。人工首次可见时间改为可选辅助参照，未记录时仍按官方起点评价，全背景或零候选的事件仍计入分母；新增相关回归测试后共18项测试通过。正式独立评价及结果复核现已完成；训练/验证标签、冻结参数和模型哈希仍一致，未因测试结果修改模型、标签或工作阈值。
