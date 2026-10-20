# 第四次实验：三个完整FIgLib火灾事件

来源：[HPWREN / UCSD FIgLib](https://www.hpwren.ucsd.edu/FIgLib/)，[官方数据索引](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html)。下载及核验日期：2026-10-20。引用和衍生成果请注明HPWREN / University of California San Diego。

| 事件 | 日期 | 官方事件目录 / 本地子目录 | 机位 | 图片数 | 原始尺寸 | JPEG字节数 |
| --- | --- | --- | --- | ---: | --- | ---: |
| Alison | 2018-06-02 | [20180602_Alison_sp-s-mobo-c](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20180602_Alison_sp-s-mobo-c/index.html) | sp-s-mobo-c | 81 | 3072×2048 | 61,757,144 |
| West | 2018-07-06 | [20180706_West_lp-n-mobo-c](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20180706_West_lp-n-mobo-c/index.html) | lp-n-mobo-c | 81 | 3072×2048 | 72,909,964 |
| Cranston | 2018-07-25 | [20180725_Cranston_hp-n-mobo-c](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20180725_Cranston_hp-n-mobo-c/index.html) | hp-n-mobo-c | 81 | 2048×1536 | 51,902,600 |

合计243张原始彩色JPEG，186,569,708字节（约186.6 MB / 177.9 MiB），不含来源和核验文件。未裁剪、增强、重新编码或改名。

每起事件均包含官方首次可见烟羽前40张、该时刻1张、后40张，范围−2400至+2400秒，严格间隔60秒。0时刻表示官方标记的首次可见烟羽，不能当作真实点火时刻。

三个事件在模型推理前选定，名称、日期、机位均互不相同，也不同于第三次实验的Whittier；各事件之间及与Whittier之间不存在相同SHA256的JPEG。此处“独立”指不同火灾事件，是否与模型预训练数据重叠尚未核实。

每个事件子目录包含：

- 81张原始JPEG：实际实验输入。
- `source_index.html`：下载时保存的官方目录快照，用于核对完整文件列表。
- `manifest.json`：每张图片的原始URL、时间戳、偏移、尺寸、字节数、采集SHA256，以及目录快照SHA256；这些校验值由本次采集计算，不是官方发布的校验和。
- `verification.json`：下载时完整解码和离线核验的记录。
- `SOURCE.md`：该事件的来源、时间和使用说明。

本次下载未纳入人工目标框或已确认的报警标签。出现前画面也可能存在其他烟雾，应人工核对模型框的位置。实验入口、复核方法及运行命令见[第四次实验说明](../../experiments/exp004_figlib_multi_event/README.md)。
