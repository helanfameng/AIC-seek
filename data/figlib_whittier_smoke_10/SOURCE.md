# FIgLib连续烟雾试跑样例

- 来源：HPWREN / University of California San Diego，FIgLib。
- 数据集说明：https://www.hpwren.ucsd.edu/FIgLib/
- 事件目录：https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20170708_Whittier_syp-n-mobo-c/index.html
- 事件：20170708_Whittier_syp-n-mobo-c，同一固定相机。
- 选取10张原始JPEG，从1499546263_+00000.jpg至1499546803_+00540.jpg，每60秒一张。
- 文件名第一个数字是原始时间戳；后缀是相对首次可见烟羽的秒数，不是真实点火时间。
- 原图3072×2048，未裁剪、增强或改名。按文件名排序即可得到时间顺序。
- 用途：预训练模型推理流程试跑。不是独立测试集，不据此声明泛化性能；与模型训练数据是否重叠未核实。
- 原始文件名不是Pyronear日期格式，其时间戳解析可能为None；顺序仍然有效。计算时间指标时须明确使用本序列60秒采样间隔。
- 使用和衍生成果须注明HPWREN来源：https://www.hpwren.ucsd.edu/ 。其他使用条件以官方说明为准。
- 下载日期：2026-10-18。
