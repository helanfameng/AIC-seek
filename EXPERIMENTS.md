# 实验记录

本文件维护各次实验的索引、主要发现与结果查看方式。根目录README负责通用安装和使用说明；完整数据和详细记录保留在各实验子目录。

## 第一次实验：官方模型GPU试跑

日期：2026-10-18。使用官方v0.6.0模型，对Whittier同一相机的10张连续图片进行整段推理。

- [实验说明](experiments/exp001_pyronear_baseline/README.md)
- [预测JSON](experiments/exp001_pyronear_baseline/predictions.json)
- 结果：判定有烟，最高保留轨迹概率0.993754，保留1条轨迹，范围为第5～10张。
- 局限：本次未计算首次在线报警时刻；概率不是准确率，单个有烟序列不能评价误报率。

下载仓库后，用浏览器打开已保存的结果：

```powershell
Start-Process .\experiments\exp001_pyronear_baseline\visualization.html
```

页面展示本次已提交结果，不会自动加载之后复跑生成的JSON。GitHub文件页不会执行HTML。复跑命令见该实验说明。

## 第二次实验：候选与逐步输入诊断

日期：2026-10-19。

- [实验说明与结果](experiments/exp002_detector_diagnostics/README.md)
- [逐步结果汇总](experiments/exp002_detector_diagnostics/summary.csv)
- [检测框与项目机制](experiments/exp002_detector_diagnostics/当前初始项目机制.md)

本样例在原版设置下，第1～4张未输出候选，第5张首次有框，第6张形成保留轨迹，第7张首次报警（首次可见烟羽后6分钟，不含计算和通信时间）。尚未进行阈值、分辨率或无火对照实验。

安装环境和下载模型后，从仓库根目录运行：

```powershell
$RunId = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunRoot = Join-Path '.\experiments\exp002_detector_diagnostics\outputs' $RunId
& .\.venv\Scripts\python.exe .\experiments\exp002_detector_diagnostics\scripts\diagnose_sequence.py `
    --sequence .\data\figlib_whittier_smoke_10 `
    --model .\models\v0.6.0\model.zip `
    --output-dir $RunRoot `
    --device cuda
Start-Process (Join-Path $RunRoot 'visualization.html')
```

也可使用现有环境解释器和模型的完整路径。脚本拒绝写入非空目录，重跑时请指定新的输出目录。查看已提交结果时，用浏览器打开 `experiments/exp002_detector_diagnostics/visualization.html`。每个前缀只使用当时已有图片重建轨迹和裁剪；不足6张时仍保留官方重复帧填充，重复帧不代表独立观测。

## 第三次实验：Whittier完整81帧回放

状态（2026-10-20核验）：六组均已完成，两组10帧、四组81帧，共344个判断时刻。原版完整回放首次阳性为+6分钟；低阈值、高分辨率、联合设置均为+5分钟，四组出现后首次候选仍为+4分钟。四组在目标烟羽出现前均无阳性窗口，但人工审核表尚未填写，不能据此声明误报率或泛化性能。

- [详细执行方案与命令](experiments/exp003_whittier_full_replay/README.md)
- [实际结果分析与后续方向](experiments/exp003_whittier_full_replay/analysis_20261020/RESULTS.md)
- [完整数据来源](data/figlib_whittier_full_81/SOURCE.md)
- 回放入口：`experiments/exp003_whittier_full_replay/scripts/replay_sequence.py`；分阶段入口位于同目录的 `run_whittier_experiments.ps1`；自动检查位于该实验的 `tests/`。
- 先复现旧10帧，再固定单图检测方式比较10帧与完整81帧；每次使用当前及过去最近最多20张。之后进行检测阈值和分辨率对照。
- 结果写入新的`experiments/exp003_whittier_full_replay/outputs/exp003_<RunId>/`，拒绝覆盖既有目录；出现前阳性和首次正确报警需人工复核。
- 本次结果目录：`experiments/exp003_whittier_full_replay/outputs/exp003_20261019_224724`。迁移前的核算验证CSV/JSONL与配置一致、无未来帧或截断，并独立重算174条保留轨迹概率；原10帧结果精确复现，完整历史未改变原版早期判断。分析结果报告已同步当前路径，其他分析辅助文件暂留本地。

## 第四次实验：三个独立FIgLib事件的完整回放

状态（2026-10-20核验）：Alison、West、Cranston共12组81帧回放已全部完成，972步记录与汇总一致，834条轨迹概率已独立重算。Alison原版+39分钟、高分辨率+34分钟才出现目标轨迹阳性；Cranston均为+10分钟；West的1024两组存在出现前持续背景阳性，目标轨迹首次跨阈为原版+6、低阈值+7、高分辨率/联合+5分钟。原始自动整窗阳性不能直接当作首次正确预警。

- [实验说明与完整启动命令](experiments/exp004_figlib_multi_event/README.md)
- [实际结果与深入分析](experiments/exp004_figlib_multi_event/RESULTS.md)
- 时间均相对官方首次可见烟羽：完整第41张为0分钟，第42张为+1分钟，第43张为+2分钟。
- [数据来源、事件目录与核验说明](data/figlib_exp004_three_events/SOURCE.md)
- 固定第三次的模型和因果回放代码，三个事件各比较原版、低检测阈值、高分辨率及联合设置，共12组完整回放、972个判断时刻；不再重复两组10帧实验。
- 本次脚本、协议和检查位于`experiments/exp004_figlib_multi_event/`；结果运行时写入其`outputs/exp004_<RunId>/<事件>/<设置>/`，拒绝覆盖已有目录。
- 用于检验Whittier的早期候选/轨迹/判定瓶颈是否重复出现；是否与预训练数据重叠尚未核实，自动输出仍需人工复核，不作为总体误报率或模型上限的结论。

## 第五次实验：早期薄烟、轨迹关联与校准的原因定位

状态（2026-10-21）：完整正式批次`exp005_20261021_173723`已完成，12个任务均为`completed`，无时间或数量截断；共2247条记录、244次新的分类器推理。任务执行约84秒，不含前置模型加载。此前小范围试跑与正式结果分开保存。

- [实验设计、输出解释与完整启动命令](experiments/exp005_cause_localization/README.md)
- [实际结果、裁剪解释、问题判断与11篇论文技术选择](experiments/exp005_cause_localization/RESULTS.md)
- [本次完整正式输出](experiments/exp005_cause_localization/outputs/exp005_20261021_173723/)
- 直接读取第四次保存的框和轨迹，以及三个原始VIA标注；不重新运行检测器，不依赖此前未发布的标注检查辅助文件。
- Alison/Cranston分别固定已有轨迹做裁剪对照、用人工位置做独立诊断；West先固定检测框比较关联，再固定轨迹特征检查N的校准作用。
- 默认12个任务；保留原版和高分辨率来源对照，校准另包含低阈值及联合来源。人工位置和校准反事实不作为自动检测成绩。
- 实际结果：Alison对裁剪敏感，高分辨率来源人工范围/上下文2.0在+15、+16分钟跨阈；Cranston早期裁剪变体仍未跨阈。West原版替代关联将目标跨阈由+6提前到+5，但背景仍连续72个窗口阳性。固定目标证据、N由2增至4可将p从0.877628压至0.684744。
- 后续优先做EARL背景预测/局部残差的单模块对照，SmokeCPC为第二选择；关联与校准单独验证。尚未实现或验证新模型，三个开发事件不足以证明通用最优方案。
- 结果写入新的`experiments/exp005_cause_localization/outputs/exp005_<RunId>/`，拒绝覆盖已有任务。试跑与正式结果使用不同RunId。

## 第六次实验：冻结 Pyronear 的背景预测与局部残差融合

状态（2026-10-22）：baseline/seed_42正式独立评价及深入复核已完成。五种方法均检出8/10个测试事件；原版、均值残差、学习残差的检出事件中位延迟分别为4.49、3.50、2.98分钟，正常9.974摄像头小时分别产生0、1、5段误报。学习残差相对均值残差仅在Thorn额外提前，正常预测误差在验证/测试上均高于均值参考；当前尚未证明综合优势。1536及43/44种子尚无正式结果。

口径补充：8/10是冻结清单的原始程序结果。Junction的可见目标烟雾尚待完整序列确认；ml-w经用户指出左侧器械遮挡下有烟，而现有候选位于右侧。二者不能直接视为同类漏检，也不能为提高成绩而删除困难正例。当前测试已经用于诊断和方案设计，后续最终泛化结论需要新的隔离测试集。详见[问题汇总](RESEARCH_ROADMAP.md)。

- [方案、数据划分、标注方式和完整启动命令](experiments/exp006_background_residual/README.md)
- [正式结果与深入分析](experiments/exp006_background_residual/analysis_20261022/RESULTS.md)
- [冻结阈值的原始测试汇总](experiments/exp006_background_residual/outputs/exp006_main_20261021/baseline/seed_42/RESULTS.md)
- 官方检测器、关联、裁剪、DINOv2和原时序分类器全部冻结，仅训练新增预测器和融合头。借鉴EARL正常特征预测/残差，不宣称完整复现或已成立创新。
- 对照包括原版、仅校准、相同历史的普通融合、简单均值残差及学习预测残差；保留0.10/1024和0.10/1536设置。
- 按站点隔离train/val/test，排除旧开发站点；阈值只由验证集确定，独立测试同时比较延迟、漏检、误报次数和超阈时长。
- GPU兼容性试跑使用旧Alison事件8帧，成功提取8×16×384局部特征并完成原推理；不是完整实验结果。

## 第七次实验分支：PSL思想的区域分类适配

状态（2026-10-24）：标注及新版10例资格已确认；正式批次`exp007_main_20261023`完成42/43/44三个种子的18个模型、验证阈值冻结、旧方法复评及诊断评价。各方法最终均检出10/10；原版平均延迟5.10分钟、5分钟召回80%、正常9.974小时观察误报0，完整P5三种子平均分别为5.76分钟、60%和0.167段/小时，正常超阈占比5.85%。2019弱烟稳定提前约4～5分钟，但尚未证明总体优势。新版包含已经用于开发分析的原8例，属于修订诊断集。

- [完整结果、局限与后续诊断](experiments/exp007_psl_branch/analysis_20261024/ANALYSIS_20261024.md)
- [三种子原始汇总](experiments/exp007_psl_branch/outputs/exp007_main_20261023/baseline/RESULTS_ALL_SEEDS.md)、[逐事件明细](experiments/exp007_psl_branch/analysis_20261024/event_details.csv)

- [完整启动命令与输出说明](experiments/exp007_psl_branch/RUN.md)
- [运行协议](experiments/exp007_psl_branch/protocol.json)、[具体实现与原PSL的差异](experiments/exp007_psl_branch/METHODS.md)
- [程序验证记录](experiments/exp007_psl_branch/VALIDATION_20261023.md)

- [分支设计、对照组和评价条件](experiments/exp007_psl_branch/README.md)
- [两例数据的调整规则、候选初筛和实施顺序](experiments/exp007_psl_branch/DATA_REVISION_PLAN.md)：原数据分别保留为待核实与遮挡专项。
- [2026-10-23替代序列下载结果](experiments/exp007_psl_branch/REPLACEMENTS_20261023.md)：选定20191005 FIRE / vo-n及20200306 FIRE / ml-s，共158张原图。下载记录保留当时状态；新增119条候选及10例资格已于正式运行前完成确认，见[最终标注核查](experiments/exp007_psl_branch/ANNOTATION_CHECK_20261023.md)。
- [现有问题与整体改进路线](RESEARCH_ROADMAP.md)
- 固定原检测、关联与特征，先使用候选级标签对照普通分类头、单/多原型、BES、PUO及组合；普通监督对比学习可作扩展对照。
- 原论文是像素级分割，本分支是区域分类适配，不称为完整PSL复现；不同时混入残差、新鲜度或新检测器，以便判断独立贡献。
- 新鲜背景误报是该分支重点；无正确候选、旧证据延续分别由候选端和新鲜度实验处理。最终效果须在新的隔离数据上确认。

## 实验记录约定

每次实验在 `experiments/expNNN_描述/` 新建目录，保留简短README、实际预测JSON以及需要时生成的可视化。实验入口归入各自的 `scripts/`，新增运行结果写入各自的 `outputs/`，实验检查放入各自的 `tests/`。README写明目的、代码提交号、模型与数据、参数变化、运行命令、结果和局限。代码与依赖历史由Git管理，不重复复制环境和脚本。
