# 第四次实验：三个独立FIgLib事件的完整回放

人工标注更新（2026-10-21）：三个原始 VIA 项目已归档，共提供 43 个独立烟羽粗框；后段省略补框参考高分辨率组，不代表无烟。四组共用同一套人工标注，文件入口、空框含义及后续评价要求见 [人工标注规则与统一评价口径](ANNOTATION_PROTOCOL.md)。未重新推理，原始输出保持归档状态。

状态（2026-10-20结果核验）：**12组完整回放全部完成，共972个判断时刻**，批次为`exp004_20261020_201923`。逐步记录与汇总一致，834条轨迹概率已独立重算。Alison目标轨迹首次跨阈为原版/低阈值+39分钟、高分辨率/联合+34分钟；Cranston四组均+10分钟。West两组1024存在持续背景阳性，核对目标位置后的首次目标轨迹跨阈分别为+6、+7、+5、+5分钟，不能将整窗0分钟阳性当作正确预警。详见[深入结果分析](RESULTS.md)和[原始结果](outputs/exp004_20261020_201923/)。人工审核表仍未填写，不据此计算正式误报率或泛化准确率。运行记录提交号为`a22588ecbc3265b7d0c70a479a07ce6ab62e60b3`。

时间口径：0分钟是完整序列第41张；+1分钟为第42张，+2分钟为第43张。页面滑块初始位置为第1张（−40分钟），与出现后的第1分钟不同。

## 实验目的与数据

在Whittier之外的三起完整火灾事件上，检查早期候选缺失、轨迹形成和最终判定延迟是否重复出现，同时比较降低检测阈值、提高输入分辨率的影响。保持第三次实验的模型和回放方式，运行三个事件×四组设置，共12次81帧回放、972个判断时刻。相邻窗口高度重叠，972次判断不是972个独立事件样本。

| 事件键 | 日期 | 官方事件目录 | 原图数 | 尺寸 |
| --- | --- | --- | ---: | --- |
| alison | 2018-06-02 | [20180602_Alison_sp-s-mobo-c](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20180602_Alison_sp-s-mobo-c/index.html) | 81 | 3072×2048 |
| west | 2018-07-06 | [20180706_West_lp-n-mobo-c](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20180706_West_lp-n-mobo-c/index.html) | 81 | 3072×2048 |
| cranston | 2018-07-25 | [20180725_Cranston_hp-n-mobo-c](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20180725_Cranston_hp-n-mobo-c/index.html) | 81 | 2048×1536 |

同属[HPWREN / UCSD FIgLib](https://www.hpwren.ucsd.edu/FIgLib/)。数据放在`data/figlib_exp004_three_events/<官方事件目录>/`，合计约186.6 MB，详情见[数据来源与核验说明](../../data/figlib_exp004_three_events/SOURCE.md)。三起事件在推理前选定，名称、日期和机位不同，且与Whittier之间没有相同SHA256的JPEG；与预训练数据是否重叠仍未核实，不能据此宣称得到严格的独立测试集。

每起事件全部81帧均使用，采样间隔60秒，范围为首次可见烟羽前后各40分钟。第41帧对应0秒，是官方首次可见烟羽标记，不能解释为点火时刻。不同机位原始尺寸不同，保留原图，以相同的推理`image_size`设置比较。

## 固定条件和四组设置

| PowerShell阶段名 | 结果子目录 | 检测置信度阈值 | 检测image_size | 目的 |
| --- | --- | ---: | ---: | --- |
| FullBaseline | full_baseline | 0.10 | 1024 | 观察原版模型在新事件上的表现 |
| LowThreshold | low_threshold | 0.05 | 1024 | 检查早期候选是否被检测阈值过滤 |
| HighResolution | high_resolution | 0.10 | 1536 | 检查小烟羽是否受检测输入分辨率限制 |
| Combined | combined | 0.05 | 1536 | 检查两项调整的联合效果及代价 |

- 复用第三次的[replay_sequence.py](../exp003_whittier_full_replay/scripts/replay_sequence.py)，不另复制推理实现；代码更改会被哈希检查拦截。
- 同一Pyronear v0.6.0模型包，不训练或微调。模型SHA256：`6a328bbc933b4d92855cbcbde06a6aac165a386d08b09e9de59f7448046d455e`。
- 回放脚本SHA256（CRLF统一为LF后）：`ea9185b12e09af6670a0c2d014525f090d959e8dce97972d123c2f9b6f269c86`。
- 每次仅用当前及过去最近最多20张；`per-frame-cache`逐张检测，检测batch为1；仅用第一张已有图片预热1次。不足6帧仍保留模型包的对称重复填充，重复图片不是额外独立观测。
- 保留包内NMS IoU 0.2、轨迹参数、1.5倍裁剪上下文、224裁剪尺寸和概率校准器；最终logistic阈值保持`0.7207499167831384`。低阈值组改变的是检测阈值。
- 单GPU顺序执行12组，顺序为Alison四组、West四组、Cranston四组。省去第三次的两组10帧复现实验。
- 缓存复用第三次脚本设置的项目内`.cache/whittier_replay/`；不会把实验脚本或结果放到仓库根目录。

## 目录用途

```text
data/figlib_exp004_three_events/
  SOURCE.md
  20180602_Alison_sp-s-mobo-c/  # 81张原图及4个来源/核验文件
  20180706_West_lp-n-mobo-c/
  20180725_Cranston_hp-n-mobo-c/
experiments/exp004_figlib_multi_event/
  README.md
  events.json                  # 冻结事件、数据范围和四组参数
  scripts/
    download_events.py         # 下载/续传和离线校验
    run_experiments.py         # 生成计划、检查环境、调度回放、终端汇总
    run_experiments.ps1        # PowerShell统一入口
  tests/test_experiment.py     # 数据映射、完整分组、因果窗口及防覆盖检查
  outputs/exp004_<RunId>/      # 首次实际运行时创建
    batch.json                # 本批协议、模型、运行环境和数据清单哈希
    alison/<设置名>/
    west/<设置名>/
    cranston/<设置名>/
```

## 完整启动命令

使用现有GPU环境。以下命令在同一个PowerShell窗口依次运行；检查失败时先处理终端报错。数据已下载，日常执行无需再次联网。

```powershell
Set-Location 'D:\AIC\AIC-seek'
$Python = 'D:\AIC\pyronear_lab\.venv\Scripts\python.exe'
$Model = 'D:\AIC\pyronear_lab\models\v0.6.0\model.zip'
$Runner = 'D:\AIC\AIC-seek\experiments\exp004_figlib_multi_event\scripts\run_experiments.ps1'
$RunId = Get-Date -Format 'yyyyMMdd_HHmmss'

# 核验全部243张图片、固定模型/脚本及CUDA；不加载权重、不推理
& $Runner -Stage Check -Python $Python -Model $Model -Device cuda -RunId $RunId

# 打印12组实际命令和输出位置；不创建实验结果
& $Runner -Stage Plan -Python $Python -Model $Model -Device cuda -RunId $RunId

# 开始12组完整回放；预计共972个判断时刻
& $Runner -Stage All -Python $Python -Model $Model -Device cuda -RunId $RunId
```

本次实际运行使用`torch 2.7.1+cu118`、CUDA 11.8、NVIDIA GeForce RTX 4060 Laptop GPU。12组完整回放总历时约25分05秒，972步中单步推理最长约7.07秒，记录的峰值显存最高为572.8 MiB。具体耗时与显存取决于候选和轨迹数量，分组测量与统计口径见[结果分析](RESULTS.md)。

也可以按参数组分四次执行，四次保持同一个`$RunId`。**此方案和上面的All方案二选一**，不要先完成某组再对同一RunId执行All：

```powershell
& $Runner -Stage FullBaseline -Python $Python -Model $Model -RunId $RunId
& $Runner -Stage LowThreshold -Python $Python -Model $Model -RunId $RunId
& $Runner -Stage HighResolution -Python $Python -Model $Model -RunId $RunId
& $Runner -Stage Combined -Python $Python -Model $Model -RunId $RunId
```

每条命令依次跑三个事件。仅执行某个事件的某一组，例如：

```powershell
& $Runner -Stage FullBaseline -Event West -Python $Python -Model $Model -RunId $RunId
```

纯Python入口等价命令，亦可从任意当前目录执行：

```powershell
& $Python -X utf8 -B 'D:\AIC\AIC-seek\experiments\exp004_figlib_multi_event\scripts\run_experiments.py' run `
    --event all --stage all --model $Model --device cuda --run-id $RunId
```

## 完成后查看结果

```powershell
$RunRoot = Join-Path 'D:\AIC\AIC-seek\experiments\exp004_figlib_multi_event\outputs' "exp004_$RunId"
& $Runner -Stage Summary -Python $Python -Model $Model -RunId $RunId
Start-Process (Join-Path $RunRoot 'alison\full_baseline\visualization.html')
```

Summary仅把现有文件汇总为终端CSV，不生成辅助分析目录；空白时刻表示没有对应自动输出。完成后再次打开PowerShell，须把`$RunId`设为实际已有批次编号，避免误用新的时间戳。

每个“事件/设置”目录保留与第三次一致的六个文件：

| 文件 | 用途 |
| --- | --- |
| run.json | 状态、实际命令、代码版本/哈希、模型配置、环境及输入哈希 |
| summary.csv | 81个判断时刻的候选、轨迹、阳性、耗时和显存 |
| steps.jsonl | 原始框、轨迹、裁剪来源、校准特征等逐步证据 |
| summary.json | 自动统计的首次候选、首次保留轨迹、首次阳性等 |
| manual_review.csv | 待填写的目标烟羽/其他烟雾及候选和阳性是否正确 |
| visualization.html | 浏览器查看原图、当前候选、轨迹和裁剪范围 |

先核对12组`run.json`均为`completed`，每组`summary.json.completed_steps`为81。比较各事件的首次正确候选、首次正确轨迹/阳性与0时刻的差值，并记录高分辨率及低阈值带来的耗时、显存和额外框。图片中烟羽较小，应结合原图和框的位置填写人工复核表；自动首次候选或阳性不能直接当作正确识别。

这些结果用于判断Whittier的瓶颈是否在新事件重复出现。出现前40分钟只是有限背景片段，不能替代长时无火视频，也不能据三起火灾直接给出总体误报率或模型上限。若利用这些事件继续选参数，应将它们视为开发诊断数据，另留未参与调参的数据进行最终验证。

## 中断、重跑和数据核验

入口会在开始前检查本次请求的所有输出目录。任意一个已存在就整体报错，保留现有结果。失败组同样保留`run.json`和已写入的CSV/JSONL，不会自动删除或覆盖。

恢复时用原RunId配合`-Event`和具体`-Stage`，只运行尚未创建目录的组合。失败组目录已存在时，使用新RunId单独重跑该组；不要将两个批次混称为同一批结果。同RunId下会检查协议、模型、清单与核心源码和运行环境是否仍匹配。

仅检查原图，不导入PyTorch：

```powershell
& $Runner -Stage Verify -Python $Python
```

首次下载中断、尚未形成`manifest.json`时，可用以下命令续传，复用并核验已完成文件。已有完整事件再次执行时只核验，不覆盖；如果缺图或哈希不一致则报错，需先人工调查文件变化。

```powershell
& $Runner -Stage Download -Python $Python
```

下载使用HTTPS，Windows优先采用系统`curl.exe`，Python需要Pillow；其余平台采用Python标准库HTTPS。官方目录必须恰好匹配81帧才开始下载，图片完整解码后才接纳为原图。下载临时文件仅放在对应的`data`子目录。

检查本次调度与数据协议：

```powershell
& $Python -X utf8 -B -m unittest discover `
    -s 'D:\AIC\AIC-seek\experiments\exp004_figlib_multi_event\tests' `
    -p 'test_*.py' -v
```

测试不运行模型推理；已有逐张图像完整解码与哈希检查由`Verify`/`Check`执行。已完成并核验的`outputs/exp004_20261020_201923/`批次随本次发布归档；其他批次继续沿用仓库的`outputs/`忽略规则，后续发布时再明确纳入。
