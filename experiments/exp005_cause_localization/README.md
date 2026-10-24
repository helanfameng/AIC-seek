# 第五次实验：早期薄烟、轨迹关联与校准的原因定位

状态（2026-10-21）：**完整第五次实验已完成**。正式批次为 [exp005_20261021_173723](outputs/exp005_20261021_173723/)，12 个任务均为 `completed`，未设置时间截断或数量限制；共 2247 条记录、244 次新的分类器推理。任务执行时间为北京时间 17:44:42～17:46:06，约 84 秒，不含前置模型加载。此前 243 张输入图/环境检查、9 项针对性测试及 +4～+6 分钟试跑已通过；试跑不并入正式分析。

**先读 [RESULTS.md：实验设置、实际结果、问题判断与论文技术选择](RESULTS.md)。** 其中用简单语言说明了“裁剪”是什么，并区分已证实机制、跨事件研究假设及尚未验证的改进。

| 阶段 | 实际结果概要 |
| --- | --- |
| Crop 裁剪 | Alison 调整人工范围/上下文可明显提分，高分辨率来源 +15、+16 分钟在人工诊断中跨阈；Cranston 的 +2～+5 分钟全部变体仍未跨阈 |
| Oracle 人工位置 | 正确烟区下分数仍随上下文和时间变化，Cranston 最早期证据仍弱；只报告原始 L |
| Association 关联 | West 原版替代关联使目标首次跨阈从 +6 提前至 +5，最高分目标时刻仍为 +7，背景观察区仍连续 72 个窗口阳性 |
| Calibration 校准 | 固定目标证据，N 从 2 增至 4 可令 p 从 0.877628 降至 0.684744，解释低阈值来源的一处退步；尚未重新拟合校准器 |

人工位置诊断、轨迹跨阈和校准反事实均不作为实际新告警或泛化成绩。下一步优先验证 EARL 的正常背景预测/局部残差机制，SmokeCPC 为第二选择；这是技术选型建议，尚未实现或验证。

查看**已完成批次**无需重跑：

```powershell
& 'D:\AIC\AIC-seek\experiments\exp005_cause_localization\scripts\run_experiment.ps1' `
  -Stage Summary -Python 'D:\AIC\pyronear_lab\.venv\Scripts\python.exe' `
  -RunId '20261021_173723'
```

本轮复用第四次已归档的检测框和轨迹，不重新运行 YOLO，不训练或修改原始模型。人工框直接读取三个原始 VIA 项目，遵循 [人工标注规则](../exp004_figlib_multi_event/ANNOTATION_PROTOCOL.md)，不依赖此前未发布的检查脚本、标签导出、notebook 或仓库外某个源码文件的硬编码路径。运行时使用现有 Python 环境已安装的 `temporal_model` 包。

## 1. 实验具体改变什么

| 阶段 | 事件与默认来源 | 固定条件 | 对照变化 |
| --- | --- | --- | --- |
| Crop：已有轨迹裁剪 | Alison、Cranston；原版及高分辨率，共 4 个任务 | 原轨迹帧、有效槽位、权重、224 输入、T/C/N、校准器 | 原裁剪；仅改人工中心；再单独改裁剪范围或上下文 |
| Oracle：人工位置诊断 | Alison、Cranston，各 1 个任务 | 同一时刻的人工可见历史、人工中心、权重、224 输入 | 上下文系数 1.0、1.5、2.0；只报告原始 L |
| Association：关联诊断 | West；原版及高分辨率，共 2 个任务 | 第四次检测框、过滤/合并规则、分类器、校准器 | 原 IoU=0.20；IoU=0.10；中心位移/尺度辅助关联 |
| Calibration：校准反事实 | West；第四次全部四组，共 4 个任务 | 已保存的每条轨迹、L/T/C、校准系数和阈值 | 原 N；N=1；额外增加 1 或 2 条无关轨迹 |

默认总计 **12 个任务**。`-Stage All` 使用上表默认来源设置；`-Setting All` 是额外选择，会把 Crop 和 Association 扩展到第四次全部四种来源设置，首次正式运行无需加它。

### Crop 的五个变体与单项对照

| 变体 | 改动 | 应比较的参考 |
| --- | --- | --- |
| `original` | 完整复现保存轨迹的原裁剪 | 第四次保存分数；偏差超过 0.001 即停止 |
| `manual_center_original_extent` | 只把中心移至人工范围中心，原宽高及上下文不变 | `original` |
| `manual_extent_c1p5` | 固定人工中心，改为人工范围，仍取上下文 1.5 | `manual_center_original_extent` |
| `manual_extent_c1` | 固定人工中心及范围，只将上下文改为 1.0 | `manual_extent_c1p5` |
| `manual_extent_c2` | 固定人工中心及范围，只将上下文改为 2.0 | `manual_extent_c1p5` |

人工范围取**当前及过去、原轨迹实际输入槽位中已有人工框的并集**，不引入未来标注，不增加原轨迹帧。每个变体明确写出 `contrast_reference`、`changed_component` 和相应分数差值。跨不同参考的差值不要混称为同一个单因素效果。

Crop 只选择当前真实检测框与人工粗范围重叠、且唯一的保留轨迹；无保留轨迹或归属有歧义则记为跳过，不自动补框。这个选择是定位诊断，不是经过独立审核的正式真阳性身份。

Oracle 是另一套输入条件：每个时刻使用最近最多 20 个连续、人工可见且有框的历史帧；至少 2 个独立帧才运行。它不受检测候选是否存在限制。人工中心和范围也仅由当前及过去这些框确定。因为没有对应的检测 C 和保留轨迹数 N，**不人为填入 C=1、N=1 来制造校准概率**，只报告 L；不得与 Crop 合并计算自动检测提前量。

### West 关联与校准

关联阶段三种策略的检测输入完全相同。中心辅助策略优先保留 IoU≥0.20 的连接；低于门槛时，仅允许中心距离不超过两框较大对角线的 0.75 倍、宽高尺度比均不超过 2.5 的同类别连接。仍采用原贪心一对一分配、缺失容忍、过滤、合并和插值，详见 [protocol.json](protocol.json)。这是待检验的诊断策略，不预设它优于原策略。

关联变化自然可能引起 T/C/N 和最终概率变化，因此同时保存原始 L、轨迹特征及完整连接。分类器输入与第四次完全相同时可复用已保存的 L；输入发生变化才重新分类，来源写入 `score_source`。原关联策略必须重建出第四次相同的轨迹与窗口概率。

校准阶段不调用 GPU，不重新分类，只重算同一轨迹在不同 N 下的概率。`N=1` 或添加无关轨迹仅用于解释耦合，不直接作为新的部署校准器，也不重新拟合系数。

West 的背景观察区为原结果分析中 `(180,1130)` 至 `(780,1500)` 的山地范围，**它不是完整背景真值**。程序保存该观察区内轨迹的持续阳性变化，不据此给出全局误报率。West 0～+2 分钟肉眼不可辨识、后段省略补框等状态均保留，未将空框当无烟。

## 2. 正式重跑方式（使用新的 RunId）

本次正式实验已完成，下面保留完整重跑命令。重跑必须使用新的 RunId；仅查看既有结果请使用上方固定批次的 Summary 命令。在同一个 PowerShell 窗口按顺序执行，使用已经跑通第四次实验的环境，无需激活环境、下载数据或新增 Python 依赖。

**第一步：设置路径和本次运行编号。**

```powershell
Set-Location 'D:\AIC\AIC-seek'
$ErrorActionPreference = 'Stop'
$Python = 'D:\AIC\pyronear_lab\.venv\Scripts\python.exe'
$Model = 'D:\AIC\pyronear_lab\models\v0.6.0\model.zip'
$Runner = 'D:\AIC\AIC-seek\experiments\exp005_cause_localization\scripts\run_experiment.ps1'
$RunId = (Get-Date).ToUniversalTime().AddHours(8).ToString('yyyyMMdd_HHmmss')
```

**第二步：检查输入和运行环境，并确认实际任务清单。**

```powershell
& $Runner -Stage Check -Python $Python -Model $Model -Device cuda -RunId $RunId
& $Runner -Stage Plan -Python $Python -Model $Model -Device cuda -RunId $RunId
```

Check 核对原图、VIA、第四次保存输出、固定模型和已安装核心包版本，并检查 CUDA；不加载模型权重、不推理、不创建实验结果。依赖导入可能建立 `D:\AIC\AIC-seek\.cache\exp005\` 下的运行缓存。Plan 只打印 12 个任务、时段和路径。检查失败时先处理报错。

**第三步：一次完成正式实验。**

```powershell
& $Runner -Stage All -Python $Python -Model $Model -Device cuda -RunId $RunId
```

这条命令依次执行全部 12 个任务，GPU 模型只加载一次。不要添加 `-OffsetMin`、`-OffsetMax` 或 `-Limit`，以保留完整正式范围。每个任务会打印进度；本次正式任务执行约 84 秒，不含前置模型加载，其他运行的耗时取决于需要重新分类的轨迹数量及实际设备。

**第四步：查看汇总与结果目录。**

```powershell
& $Runner -Stage Summary -Python $Python -RunId $RunId
$ResultRoot = Join-Path 'D:\AIC\AIC-seek\experiments\exp005_cause_localization\outputs' "exp005_$RunId"
Write-Host $ResultRoot
```

重新打开 PowerShell 时，把 `$RunId` 设为实际已完成的编号，不要重新生成时间戳来查旧结果。

## 3. 分阶段启动方式

如果需要分步执行，下面四条命令**替代**上面的 All，保持同一个 `$RunId`。不要先跑其中一项，再对同一编号执行 All。

```powershell
& $Runner -Stage Crop -Python $Python -Model $Model -Device cuda -RunId $RunId
& $Runner -Stage Oracle -Python $Python -Model $Model -Device cuda -RunId $RunId
& $Runner -Stage Association -Python $Python -Model $Model -Device cuda -RunId $RunId
& $Runner -Stage Calibration -Python $Python -Model $Model -Device cuda -RunId $RunId
```

仅运行一个事件、一个来源设置，例如 Alison 原版裁剪：

```powershell
& $Runner -Stage Crop -Event Alison -Setting Full_Baseline `
  -Python $Python -Model $Model -Device cuda -RunId $RunId
```

仅 West 原版关联：

```powershell
& $Runner -Stage Association -Event West -Setting Full_Baseline `
  -Python $Python -Model $Model -Device cuda -RunId $RunId
```

Oracle 只有 Alison/Cranston 两个事件，不按检测设置重复；Calibration 默认包含四种来源设置。用 `-Setting Low_Threshold` 可以单独查看低阈值来源，但这仍然只改变所读取的第四次输出，不重新跑检测器。

## 4. 小范围试跑与 CPU 入口

以下为另一个运行编号的试跑，不能当作完整实验。时间参数单位为**相对官方起点的秒**，240～360 即 +4～+6 分钟；只筛选输出时刻，保留其之前所需的历史窗口。

```powershell
$SmokeId = "smoke_" + (Get-Date).ToUniversalTime().AddHours(8).ToString('yyyyMMdd_HHmmss')
& $Runner -Stage All -Python $Python -Model $Model -Device cuda `
  -RunId $SmokeId -OffsetMin 240 -OffsetMax 360
& $Runner -Stage Summary -Python $Python -RunId $SmokeId
```

纯校准可用 Python 标准库单独运行，不需要模型包、PyTorch 或 CUDA。下面的 `$CpuRunId` 与正式 GPU 批次分开：

```powershell
$CpuRunId = "calibration_cpu_" + (Get-Date).ToUniversalTime().AddHours(8).ToString('yyyyMMdd_HHmmss')
python -X utf8 -B 'D:\AIC\AIC-seek\experiments\exp005_cause_localization\scripts\run_experiment.py' run `
  --stage calibration --event west --device cpu --run-id $CpuRunId
```

完整分类器也支持 `-Device cpu`，但耗时可能明显增加。默认推荐沿用已核验的 CUDA 环境。

## 5. 输出文件及怎么读

正式范围为 Crop 的人工有框时段（Alison 0～+16、Cranston 0～+5 分钟）、Oracle 的同一人工时段，以及 West 关联/校准的完整 −40～+40 分钟。

每个批次只有一个 `batch.json`，每个任务固定四个文件：

| 文件 | 用途 |
| --- | --- |
| `run.json` | 完成/失败状态、命令、来源哈希、程序/协议版本及实际环境 |
| `records.jsonl` | 逐时刻、逐变体的完整证据，包括输入帧、裁剪或关联细节 |
| `summary.csv` | 逐条记录的表格版本，列表以 JSON 文本保存；完整结构以 JSONL 为准，不是“一任务一行”的总成绩表 |
| `summary.json` | 该任务的跳过原因、变体统计和限定范围内的诊断汇总 |

默认子目录为：

```text
outputs/exp005_<RunId>/
  batch.json
  crop/{alison,cranston}/{full_baseline,high_resolution}/
  oracle/{alison,cranston}/manual/
  association/west/{full_baseline,high_resolution}/
  calibration/west/{full_baseline,low_threshold,high_resolution,combined}/
```

本次正式批次正好包含 49 个文件：1 份 `batch.json` 和 12×4 份任务文件。`no_kept_target_overlap`、`insufficient_visible_history`、`no_kept_tubes` 是样本处理状态，不表示任务运行失败；任务完成状态看 `run.json`。

先核对 12 份 `run.json` 都为 `completed`，再看：

1. **裁剪是否影响分类：** 以 `contrast_reference` 为配对参考，比较 `delta_logit_vs_contrast`、`delta_probability_vs_contrast`、覆盖率和目标在 224 输入中的尺寸。`*_in_patch_unclipped` 是人工粗范围按缩放比例投影的尺寸，越界部分是否被裁掉还需结合覆盖率。
2. **给定人工位置是否仍低分：** Oracle 固定时序帧比较上下文；与 Crop 分开解释。原始 L 的符号或变化不能直接替代完整校准后的告警结论。
3. **关联是否丢失早期连续目标：** 看 West +4/+5/+6 分钟的原始框、连接、过滤后轨迹、独立真实观测数、L 与 p。同时看背景观察区是否产生更多或更持续的轨迹。
4. **校准是否因其他轨迹压低目标：** 看 West +6 分钟原版与低阈值来源的 `frozen_L/T/C`、`source_N`，以及不同 N 的概率。背景观察区也要检查，避免只强调目标概率提升。

关联汇总中的首次人工范围重叠/跨阈，限于有人工框且被选中的时刻，不是正式首次正确告警。正式漏检率、相近误报水平下的提前量及新事件泛化评价留待相应真值和独立数据齐备后进行。

## 6. 中断与重跑

程序拒绝覆盖已有任务目录；开始前会检查本次请求的全部输出路径。若已有任一目录，整次请求立即停止。相同 RunId 还会核对程序、协议、模型、设备和筛选范围，防止混入不同实验。

- 尚未创建的任务，可用原 RunId 加 `-Stage/-Event/-Setting` 精确选择后续执行。
- 已失败并留下目录的任务，使用**新 RunId**重跑该组合；保留旧结果，不删除或覆盖。
- 试跑和正式实验使用不同 RunId，正式实验不附带时间筛选或数量限制。
- 原图、VIA、第四次输出、已安装模型核心源码均只读；本轮输出和运行缓存都在本仓库 D 盘目录内。

针对性测试入口（不加载模型或运行 YOLO）：

```powershell
& $Runner -Stage Tests -Python $Python
```
