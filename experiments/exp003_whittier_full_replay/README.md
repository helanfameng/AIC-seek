# exp003执行方案：Whittier完整81帧回放

状态更新（2026-10-20）：**六组模型推理均已完成，结果位于本实验目录下的[outputs/exp003_20261019_224724](outputs/exp003_20261019_224724/)。** 实测结论与后续方向见[结果分析](analysis_20261020/RESULTS.md)。本页保留数据准备和执行方案。

所有新增数据、缓存和输出均放在D盘项目目录；原exp001、exp002和原10帧保留。下述命令和预检说明是本次实验的执行记录，参数效果以实际结果文件为准。

## 目录归档

- `scripts/`：下载器、时间回放入口和六组实验的PowerShell执行脚本。
- `outputs/exp003_<RunId>/`：每个运行批次的完整原始结果，六个组目录及其文件均保留。
- `tests/test_replay_sequence.py`：7项无需模型推理的回放检查。
- `analysis_20261020/RESULTS.md`：发布本次结果分析，路径引用已同步；其余分析辅助文件保留本地，不随本次提交发布。

2026-10-20目录迁移只调整入口位置、项目根目录定位和结果页面的图片链接。原始预测、配置、人工审核表与历史命令保持不变；`run.json`里的路径和脚本哈希描述当时的运行，不能改写为迁移后版本。现行复跑命令已更新如下。共享输入仍位于项目根目录的 `data/`，缓存仍位于 `.cache/whittier_replay/`。

从仓库根目录运行检查（解释器亦可替换为其他可用Python）：

```powershell
& 'D:\AIC\pyronear_lab\.venv\Scripts\python.exe' -B -m unittest discover `
    -s 'D:\AIC\AIC-seek\experiments\exp003_whittier_full_replay\tests' `
    -p 'test_*.py' -v
```

## 已准备的数据与环境

- 数据目录：`D:\AIC\AIC-seek\data\figlib_whittier_full_81`。
- 来源：[HPWREN / UCSD FIgLib](https://www.hpwren.ucsd.edu/FIgLib/)；[官方事件目录](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20170708_Whittier_syp-n-mobo-c/index.html)。
- 81张原始JPEG，3072×2048，每60秒一张；从`1499543863_-02400.jpg`到`1499548663_+02400.jpg`，合计67,185,389字节（约64.1 MiB）。
- 第41张`1499546263_+00000.jpg`为官方首次可见烟羽，前40张位于出现前，后40张位于出现后。0秒不是真实点火时间。
- 81张已逐图完整解码；下载时计算的SHA256、尺寸及来源URL在`manifest.json`；这些哈希是本次采集的校验记录，不是官方发布的校验和。
- 与旧10帧对应的文件SHA256全部一致，见`verification.json`。原始图像未裁剪、增强或改名。
- 现有解释器：`D:\AIC\pyronear_lab\.venv\Scripts\python.exe`。
- 现有模型：`D:\AIC\pyronear_lab\models\v0.6.0\model.zip`。
- 模型SHA256：`6a328bbc933b4d92855cbcbde06a6aac165a386d08b09e9de59f7448046d455e`。
- 本轮环境预检确认：PyTorch `2.7.1+cu118`，CUDA运行时11.8，CUDA可用，GPU为RTX 4060 Laptop GPU；时序模型代码来自`D:\AIC\pyronear_lab\temporal-model\core\src`。
- 已通过7项不涉及新模型推理的检查：81个因果窗口、重复帧独立观测数、历史轨迹裁剪显示、出现前后阳性分离、真实输入dry-run、旧结果目录保护、现有5个真实预测的校准得分重算；PowerShell脚本语法检查通过。

## 实验协议

1. 81张按时间依次到达，每次只使用当前及过去最近最多20张；一分钟采样下，20张的首尾时间跨度为19分钟。
2. 所有窗口独立重建轨迹、合并、裁剪、分类和校准；轨迹ID不跨窗口延续，原20帧上限保持不变。
3. 保留官方不足6帧时的symmetric重复填充；报告有效槽数、填充数和每条轨迹的独立观测数，不把重复帧当新证据。
4. 正式完整回放默认`per-frame-cache`：每张新图只检测一次，检测批大小为1，再复用历史检测结果；未来帧不会提前检测。
5. 额外提供`window-batch`用于旧10帧复现：在每个前缀重新成批检测当时已有图像。它与旧脚本的批次方式一致，但耗时不能当部署中每帧仅检测一次的耗时。
6. 固定权重、NMS IoU 0.2、轨迹条件、1.5倍上下文、224×224裁剪和logistic报警阈值约0.7207499168。后续四组只调整检测阈值和检测尺寸。
7. 对每个时刻保存所有候选和保留轨迹，分解校准特征L/T/C/N及线性项；不启用上游内部首次触发搜索，首次阳性由回放各时刻结果取得。
8. 仅用最早可见的一张图进行一次预热，不使用未来烟雾图。首次真正进入分类器时仍可能包含分类器冷启动开销；耗时用于本协议内比较，不冒充完整部署延迟。

## 1．设置变量并预检

在PowerShell中依次执行。下面的绝对路径使用本机已经成功运行的环境，无需新装环境或下载模型。

```powershell
Set-Location 'D:\AIC\AIC-seek'
$Python = 'D:\AIC\pyronear_lab\.venv\Scripts\python.exe'
$Model = 'D:\AIC\pyronear_lab\models\v0.6.0\model.zip'
$Sequence = 'D:\AIC\AIC-seek\data\figlib_whittier_full_81'
$RunId = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunRoot = Join-Path 'D:\AIC\AIC-seek\experiments\exp003_whittier_full_replay\outputs' "exp003_$RunId"

& .\experiments\exp003_whittier_full_replay\scripts\run_whittier_experiments.ps1 -Stage Check -Python $Python -Model $Model
```

应看到`verified_images: 81`、`old_images_identical: 10`、`selected_images: 81`、`uses_future_frames: false`、`CUDA available: True`及预检通过。此步骤不加载模型权重，不运行推理。

脚本将当前进程的临时文件和运行缓存设到`D:\AIC\AIC-seek\.cache\whittier_replay`。若重新开启PowerShell，重新执行本节变量定义；要继续写入同一批实验目录，手工把`$RunId`设回之前的值。

## 2．先复现原10帧的逐步判断

```powershell
& .\experiments\exp003_whittier_full_replay\scripts\run_whittier_experiments.ps1 -Stage Cold10Legacy -RunId $RunId -Python $Python -Model $Model
Start-Process (Join-Path $RunRoot 'cold10_legacy\visualization.html')
Import-Csv (Join-Path $RunRoot 'cold10_legacy\summary.csv') | Format-Table offset_seconds,current_candidate_count,kept_tube_count,is_positive,max_probability
```

本组只用完整数据中的0～540秒共10张，采用`window-batch`。原exp002参照是：+240秒首次有候选，+300秒保留轨迹概率约0.67618，+360秒首次阳性、概率约0.83877。上述参照不是本轮新结果；应以新输出验证，GPU浮点细节可能存在轻微差异。若阶段或判断明显不同，先核查环境、代码和参数，再解释后续实验。

本组`source_frame_1based`是完整81帧内的序号41～50，`step_1based`是本次回放序号1～10。比较延迟请用`offset_seconds`，不要混用两个序号。

## 3．固定单帧检测方式，建立10帧参照

```powershell
& .\experiments\exp003_whittier_full_replay\scripts\run_whittier_experiments.ps1 -Stage Cold10Cached -RunId $RunId -Python $Python -Model $Model
```

本组范围仍为0～540秒，检测改为每张一次、批大小1。先比较`cold10_legacy`与`cold10_cached`，观察批次变化是否影响分数或跨阈。之后用`cold10_cached`和完整回放比较历史输入作用，避免把检测批次差异归因于监控历史。

## 4．完整81帧原版基线（优先完成）

```powershell
& .\experiments\exp003_whittier_full_replay\scripts\run_whittier_experiments.ps1 -Stage FullBaseline -RunId $RunId -Python $Python -Model $Model
Start-Process (Join-Path $RunRoot 'full_baseline\visualization.html')
Get-Content (Join-Path $RunRoot 'full_baseline\summary.json') -Raw
```

检查是否有81个输出时刻、范围−2400～+2400秒、最大真实窗口20张；对比+0～+540秒的`full_baseline`与`cold10_cached`，并人工查看出现前阳性及后续判断中断。不要将81张一次性传入旧`run_sequence.py`或`diagnose_sequence.py`。

## 5．人工核查基线，再做三组参数对照

| 阶段参数 | 输出目录名 | 检测阈值 | 输入尺寸 | 数据范围/检测方式 |
|---|---|---:|---:|---|
| `FullBaseline` | `full_baseline` | 0.10 | 1024 | 全81帧、每图一次 |
| `LowThreshold` | `low_threshold` | 0.05 | 1024 | 同上 |
| `HighResolution` | `high_resolution` | 0.10 | 1536 | 同上 |
| `Combined` | `combined` | 0.05 | 1536 | 同上 |

```powershell
& .\experiments\exp003_whittier_full_replay\scripts\run_whittier_experiments.ps1 -Stage LowThreshold -RunId $RunId -Python $Python -Model $Model
& .\experiments\exp003_whittier_full_replay\scripts\run_whittier_experiments.ps1 -Stage HighResolution -RunId $RunId -Python $Python -Model $Model
& .\experiments\exp003_whittier_full_replay\scripts\run_whittier_experiments.ps1 -Stage Combined -RunId $RunId -Python $Python -Model $Model
```

建议逐组执行和检查。也可用新`$RunId`一次执行预检和全部6组：

```powershell
$RunId = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunRoot = Join-Path 'D:\AIC\AIC-seek\experiments\exp003_whittier_full_replay\outputs' "exp003_$RunId"
& .\experiments\exp003_whittier_full_replay\scripts\run_whittier_experiments.ps1 -Stage All -RunId $RunId -Python $Python -Model $Model
```

每个阶段拒绝写入已经存在的目录。重跑请换`-RunId`，不要删除旧结果。若某阶段失败，`run.json`标记失败，已完成的CSV/JSONL会保留；不要把部分输出当完整实验。

## 不经过PowerShell包装脚本的完整回放命令

下面是`FullBaseline`的等价底层命令；输出目录请选一个从未使用的新目录。Python入口也会在加载模型前将临时目录和缓存设到D盘。

```powershell
& $Python -B 'D:\AIC\AIC-seek\experiments\exp003_whittier_full_replay\scripts\replay_sequence.py' `
    --sequence $Sequence `
    --model $Model `
    --output-dir 'D:\AIC\AIC-seek\experiments\exp003_whittier_full_replay\outputs\exp003_manual_full_baseline_001' `
    --device cuda `
    --window 20 `
    --confidence 0.1 `
    --image-size 1024 `
    --detection-mode per-frame-cache `
    --warmup 1
```

单独进行无推理、无写入的输入检查：

```powershell
& $Python -B 'D:\AIC\AIC-seek\experiments\exp003_whittier_full_replay\scripts\replay_sequence.py' --sequence $Sequence --model $Model --window 20 --dry-run
```

## 输出文件及其含义

每个成功阶段在`$RunRoot\组名`下生成：

| 文件 | 用途 |
|---|---|
| `run.json` | 运行状态、命令、代码/脚本/模型哈希、实际运行core源码哈希、原始及实际配置、环境、输入哈希 |
| `summary.csv` | 每个时刻的候选、轨迹、阳性、概率、窗口及填充、检测/管线耗时、GPU峰值显存 |
| `steps.jsonl` | 完整候选、合并前轨迹、过滤原因、预测、L/T/C/N及各线性得分项；每步写入 |
| `summary.json` | 自动汇总的首次候选、首次保留窗口、首次阳性窗口、出现前/后阳性窗口数量和连续阳性段 |
| `manual_review.csv` | 人工补充目标烟雾、其他烟雾、候选位置正确性和阳性判断正确性；初始标注留空 |
| `visualization.html` | 离线查看每张原图、当前框、分类裁剪及轨迹特征；需与原数据保持相对位置 |

`max_probability`为空表示没有保留轨迹，不能填成0。`winner_observation_age_seconds`说明最高分轨迹最近一次真实检测距当前多久，避免把仅靠历史条目保持的阳性解释为当前仍检测到烟。轨迹ID仅在一个窗口内有效，不能按ID跨窗口计算告警事件。

`steps.jsonl`中的`pre_merge_filter_failures`只描述合并前轨迹的长度/条目数条件；合并前后ID不能直接一一对应，不能据此把某个合并后轨迹的结果归因于单条初始轨迹。

`summary.json`中的`first_postappearance_positive_window`只表示0秒以后第一个阳性窗口，仍需确认触发位置；它不是自动验证的“首次正确报警”。出现前阳性窗口也不能直接认定误报。人工审核时先检查是否有其他烟雾，且连续阳性窗口不能逐帧当成独立告警次数。

## 汇总各组（自动结果，人工确认前不叫正确检出率）

```powershell
Get-ChildItem -LiteralPath $RunRoot -Directory | ForEach-Object {
    $summaryPath = Join-Path $_.FullName 'summary.json'
    $metadataPath = Join-Path $_.FullName 'run.json'
    if ((Test-Path -LiteralPath $summaryPath) -and (Test-Path -LiteralPath $metadataPath)) {
        $summaryData = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
        [pscustomobject]@{
            Group = $_.Name
            Status = $metadata.status
            Steps = $summaryData.completed_steps
            FirstPostCandidateSec = $summaryData.first_postappearance_candidate.offset_seconds
            FirstPostKeptSec = $summaryData.first_postappearance_kept_window.offset_seconds
            FirstPostPositiveSec = $summaryData.first_postappearance_positive_window.offset_seconds
            PreAppearancePositiveWindows = $summaryData.preappearance_positive_windows
        }
    }
} | Format-Table -AutoSize
```

## 核查与完成标准

- 优先人工检查−2400～−60秒的候选/阳性、0～540秒的早期烟雾，以及任意新候选和阳性跳变。所有组使用同一人工判定标准。
- 报告首次正确候选、首次有效目标轨迹、首次正确目标报警，相对首次可见烟羽计秒；自动摘要只用于定位待审核时刻。
- 参数改变后，更早有框但未更早报警时，查看L/T/C/N：低分候选增多可能降低平均分或增加轨迹数，最终校准可能抵消收益。
- 所有组应使用同一设备、检测方式、窗口和预热。1536若显存不足，失败记录保留；可以显式`-Device cpu`在新RunId下运行，但CPU/GPU耗时不能直接混比。
- 耗时是本地回放的检测与预测耗时，带CUDA同步，不包含网络、通信、写文件或模型加载。预留显存会受分配器缓存影响，比较时同时看实际分配峰值；值均包含模型常驻显存，不是增量显存。
- 这81张仍是同一事件；完整回放不能替代独立火灾事件、不同天气和长时无火片段。预训练重叠情况尚未核实。

## 下载复跑与离线检查

```powershell
& $Python -B 'D:\AIC\AIC-seek\experiments\exp003_whittier_full_replay\scripts\download_figlib_whittier.py' --output-dir $Sequence --workers 4
& $Python -B 'D:\AIC\AIC-seek\experiments\exp003_whittier_full_replay\scripts\download_figlib_whittier.py' --output-dir $Sequence --verify-only
```

下载器读取并核对官方目录中的81个文件名，按官方链接对`+`编码为`%2B`。已经存在且可完整解码的文件会复用，不覆盖；损坏文件或哈希差异会报错，需单独调查，不会自动删除历史文件。使用和衍生成果注明HPWREN来源。
