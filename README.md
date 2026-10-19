# AIC-seek：林火早期烟雾识别实验

研究固定相机复杂环境下的早期烟雾检测与时序复核。目前已在个人笔记本上跑通 Pyronear 官方预训练模型；尚未训练自主改进模型，也没有建立独立测试集性能结论。

## 当前基线

YOLO提出候选 → 关联轨迹 → 裁剪局部序列 → DINOv2与时序Transformer复核。

- [首次实验记录](experiments/exp001_pyronear_baseline/README.md)
- [预测JSON](experiments/exp001_pyronear_baseline/predictions.json)
- 可视化：下载仓库后，用浏览器打开 `experiments/exp001_pyronear_baseline/visualization.html`。GitHub文件页不会执行HTML。

## 环境与安装

首次成功环境：Windows、Python 3.12.4、PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118、CUDA运行时11.8、RTX 4060 Laptop GPU（8GB）。NumPy 1.26.4。PyTorch包自带所需运行时，驱动须兼容；无需为此另装完整CUDA Toolkit。

安装Python 3.12与Git后，在PowerShell执行：

```powershell
git clone https://github.com/helanfameng/AIC-seek.git
Set-Location AIC-seek
python -m venv .venv
$Python = "$PWD\.venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
& $Python -m pip install -r requirements.txt
& $Python -m pip check
& $Python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若 `python` 不在PATH，请使用自己的Python 3.12完整路径创建环境。requirements.txt是成功运行时的依赖记录，包含固定提交的Pyronear引用；其跨机器重新安装尚未完整验证。不要与已有YOLO/AOD环境混装。

## 模型下载

从[官方v0.6.0文件页](https://huggingface.co/pyronear/temporal-model/tree/v0.6.0)下载 `model.zip`，保存为 `models/v0.6.0/model.zip`，不需要解压。权重不随本仓库上传。

本次权重SHA256：

```text
6A328BBC933B4D92855CBCBDE06A6AAC165A386D08B09E9DE59F7448046D455E
```

核对命令：

```powershell
Get-FileHash .\models\v0.6.0\model.zip -Algorithm SHA256
```

## 运行样例

仓库已包含10张原始连续图片，每分钟一张。按文件名排序，无需改名。

```powershell
$Python = "$PWD\.venv\Scripts\python.exe"
& $Python .\run_sequence.py `
    --sequence .\data\figlib_whittier_smoke_10 `
    --model .\models\v0.6.0\model.zip `
    --output .\outputs\whittier.json `
    --device cuda
Start-Process .\experiments\exp001_pyronear_baseline\visualization.html
```

可视化页面展示已提交的首次实验结果，不会自动加载新生成的 `outputs/whittier.json`。复跑输出单独保存，避免覆盖历史结果。GPU内存不足时可用 `--device cpu` 先验证接口。

## 第二次实验：候选与逐步输入诊断

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

## 第三方来源与许可

- [Pyronear temporal-model](https://github.com/pyronear/temporal-model)：上游Apache-2.0许可；本仓库通过依赖引用使用，未将其算法表述为原创。固定提交 `c0033d32b826d505029bbf6f44037b77e958a2b6`。
- [Pyronear模型](https://huggingface.co/pyronear/temporal-model)：官方模型页标注Apache-2.0；使用时同时核对其依赖和配套检测器的许可。
- [Ultralytics](https://github.com/ultralytics/ultralytics)：提供YOLO实现，采用AGPL-3.0及商业许可选项，具体使用与分发遵守其适用条款。本仓库不将第三方组件统一重新授权。
- [HPWREN / UCSD FIgLib](https://www.hpwren.ucsd.edu/FIgLib/)：10张图片来自官方公开火灾序列。依官方要求注明HPWREN来源，详见 [SOURCE.md](data/figlib_whittier_smoke_10/SOURCE.md)。图片不因放入代码仓库而改变原有使用条件。

当前仅为研究基线。单个有烟序列不能证明准确率、误报率或真实部署效果。
