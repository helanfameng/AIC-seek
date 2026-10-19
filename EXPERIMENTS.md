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
& .\.venv\Scripts\python.exe .\diagnose_sequence.py `
    --sequence .\data\figlib_whittier_smoke_10 `
    --model .\models\v0.6.0\model.zip `
    --output-dir .\outputs\exp002_rerun `
    --device cuda
Start-Process .\outputs\exp002_rerun\visualization.html
```

也可使用现有环境解释器和模型的完整路径。脚本拒绝写入非空目录，重跑时请指定新的输出目录。查看已提交结果时，用浏览器打开 `experiments/exp002_detector_diagnostics/visualization.html`。每个前缀只使用当时已有图片重建轨迹和裁剪；不足6张时仍保留官方重复帧填充，重复帧不代表独立观测。

## 实验记录约定

每次实验在 `experiments/expNNN_描述/` 新建目录，保留简短README、实际预测JSON以及需要时生成的可视化。README写明目的、代码提交号、模型与数据、参数变化、运行命令、结果和局限。代码与依赖历史由Git管理，不重复复制环境和脚本。
