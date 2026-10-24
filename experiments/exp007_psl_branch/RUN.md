# 实验七启动说明

所有命令在Windows PowerShell执行。默认使用本机已安装的实验六环境与官方模型；不需要安装旧版MMCV/MMSeg。程序自动探测能运行的Python启动器，并处理Pyronear可编辑安装的`.pth`路径。

## 1. 首次设置变量与检查

```powershell
Set-Location D:\AIC\AIC-seek
$Runner = 'D:\AIC\AIC-seek\experiments\exp007_psl_branch\scripts\run_experiment.ps1'
$RunId = 'exp007_main_20261023'

& $Runner -Stage Check
& $Runner -Stage Tests -Device cpu
```

输入依赖：原实验六的完整原图、已保存人工标签和`exp006_main_20261021/baseline`完整缓存；两段158帧替代数据；原官方v0.6.0 `model.zip`。这些大型本机输入不随代码仓库提交。缺项会列出具体路径。

若需要指定启动器和包目录，可在任一命令后追加：

```powershell
-Python 'C:\Users\HElanfameng\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -Packages 'D:\AIC\pyronear_lab\.venv\Lib\site-packages'
```

上面这一行是参数片段，需追加在`& $Runner ...`后，不能作为单独命令执行。

## 2. 一键完成数据准备与特征提取

```powershell
& $Runner -Stage Setup -RunId $RunId -Device cuda
```

Setup依次执行Prepare、Extract、Prelabel、Status：

- 创建`data/exp007_psl_branch/`，复制原人工标签状态并替换两个序列；原图使用绝对路径引用，不复制已有图片，不改实验六。
- 只为新增77＋81帧运行冻结官方模型；其他58段的候选及特征只读复用。每条缓存保留原来源、字节指纹与因果时间检查。
- 在同一RunId中组合全部候选，新增建议只参考已确认的train特征。预标不会写入正式标签。
- 正式训练与测试阶段会核对协议、源码、清单、候选和特征指纹。完整缓存可重复执行Setup复用；中断后也用同一命令继续。

原始新增提取暂存在`outputs/<RunId>__new/baseline/`；完整组合缓存在`outputs/<RunId>/baseline/`。二者都属于实验七，不是实验六目录。

## 3. 完成候选与场景资格复核

```powershell
& $Runner -Stage Review -RunId $RunId -ReviewSplit test -Port 8767
```

打开：

- `http://127.0.0.1:8767`：候选分组复核；旧8例的已有候选标签已复制，主要补两段新序列。看原图后确认烟/背景/不确定；烟还需确认是否属于目标火情。每段完成后点击“本序列候选身份复核完成”。
- `http://127.0.0.1:8767/eligibility`：对新版10例按相同标准检查整段图像，逐例确认“合格正例”；不确定或不符合范围时记录原因，程序会列出缺项，不会自动删掉来缩小分母。

原8例的场景资格需要重新检查，不自动把旧候选标签当作整段资格。两段新序列的预览裁剪不是目标真值框。精确人工首次可见为可选辅助参照；未填时统一按官方时间评价。

页面显示已保存后，终端按Ctrl+C结束复核服务。然后执行：

```powershell
& $Runner -Stage Status -RunId $RunId
```

完整启动要求`test_ready: true`。若有不合格或无法确认的事件，先按已声明标准另行修订数据版本；不能手改资格为合格来绕过问题。

## 4. 小规模真实缓存兼容性检查

```powershell
& $Runner -Stage Smoke -RunId $RunId -Seeds 42 -Device cuda
```

Smoke只读train/val，每组各最多128个候选、2轮，默认覆盖C0、P1至P5。结果位于`baseline/smoke/<身份摘要>/`，`latest.json`指向最新检查，明确标为兼容性测试，不能被正式校准或评价使用。可选加入C1：

```powershell
& $Runner -Stage Smoke -RunId $RunId -Seeds 42 -Device cuda -Variants C0,P1,P2,P3,P4,P5,C1
```

Smoke的源码、数据、种子或分组变化会使用新的身份子目录，不覆盖旧兼容性记录；正式训练仍严格要求同一RunId中的训练身份一致。

## 5. 正式一键执行

标注完成后：

```powershell
& $Runner -Stage Run -RunId $RunId -Seeds 42,43,44 -Device cuda
```

Run先检查全部复核状态，再依次执行：原五种方法冻结复评→三个种子训练C0/P1/P2/P3/P4/P5→仅验证集选阈值→诊断集评价→多种子汇总。R0官方原分数始终作为参考，不训练它。

同一命令可在中断后重新运行：已经完成、身份一致的检查点和冻结阈值会校验复用，未完成组重新从该组固定初值训练。改变代码、协议、训练/验证标签或分组集合时用新的RunId，保留旧结果。程序不在测试上重选阈值。

## 6. 分阶段执行（与Run等价）

```powershell
& $Runner -Stage Replay    -RunId $RunId -Device cuda
& $Runner -Stage Train     -RunId $RunId -Seeds 42,43,44 -Device cuda
& $Runner -Stage Calibrate -RunId $RunId -Seeds 42,43,44 -Device cuda
& $Runner -Stage Evaluate  -RunId $RunId -Seeds 42,43,44 -Device cuda
& $Runner -Stage Aggregate -RunId $RunId -Seeds 42,43,44
```

Train/Calibrate可在新增测试标签未完成时运行，因为它们只使用train/val；Replay、Evaluate和完整Run要求测试资格与候选复核完备。

可选C1完整实验在**第一次Train/Run时**统一指定`-Variants C0,P1,P2,P3,P4,P5,C1`，三种子分组必须一致；C1已实现，但不把普通监督对比学习称为PSL。

## 7. 输出位置与解读

主目录：`D:\AIC\AIC-seek\experiments\exp007_psl_branch\outputs\<RunId>\baseline\`。

| 文件/目录 | 用途 |
|---|---|
| `run.json`、`feature_sources.json`、各序列JSONL | 组合候选、特征来源、指纹和覆盖 |
| `review_suggestions.json` | 未经确认的辅助建议 |
| `legacy_replay/RESULTS.md` | 原五种方法的冻结权重、原阈值在新数据上的复评 |
| `seed_42/`、`seed_43/`、`seed_44/` | 各组检查点、训练日志、阈值、逐事件与正常误报结果 |
| `seed_*/*_prototypes.json`、`score_timing_*.json` | 原型按事件占用、余弦关系和新增头推理时间；不把头耗时冒充全系统耗时 |
| `seed_*/per_event.csv`、`test_predictions.jsonl` | 逐事件延迟、候选下界、区域竞争及逐候选分数 |
| `RESULTS_ALL_SEEDS.md`、`aggregate.json` | 多种子结果汇总 |

共享人工状态在`data/exp007_psl_branch/labels.json`与`eligibility.json`；首次会话修改前保留备份。数据选择在本实验`data_selection.json`，训练设置在`protocol.json`。

该轮是修订诊断集，正常验证/测试仍约8/10小时；扩充正常数据及新的最终测试属于下一阶段，不把当前结果称为长期部署证明。PSL分支不同时加入证据年龄、SAHI或新编码器，独立贡献明确后再做组合。

算法的具体定义及与作者实现的差异见[METHODS.md](METHODS.md)。
