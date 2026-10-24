# 实验七标注与启动条件核查（更新：2026-10-24）

结论：训练、验证、测试所需标注和10个测试事件资格均已就绪。正式启动前最后一次GPU/模型检查结果见本报告末尾。

## 用户确认及已保存的更正

- 2019-10-05：从0分钟可见极微弱烟，+6、+7分钟相对明显。首次可见起点为0分钟；0、+1分钟且覆盖目标烟的候选已更正为烟雾。
- 其余9个场景：用户在对话中明确确认可以辨认对应火情烟雾，已保存为eligible。
- 两个替代序列共119个候选：75个烟雾、41个背景、3个不确定。2020序列+1、+2、+3分钟的三个不确定标签保留。
- 全测试集812个候选已有复核记录；所有10个火情事件已标记复核完成。

## 完整性与启动状态

- train_ready=true；test_ready=true；资格eligible=10；待处理门槛为0。
- 训练/验证标签与此前已验证程序自检的指纹一致；原实验六标签及其余4份来源文件哈希均未变化。
- 2748条候选记录与全部引用特征文件的哈希校验通过；程序代码指纹与已验证版本一致。
- 本次只更新其余9例的资格记录；未再修改候选标签、首次可见时间；未启动正式训练。

## 正式启动命令

沿用已准备好的RunId与特征缓存。PowerShell运行：

```powershell
& 'D:\AIC\AIC-seek\experiments\exp007_psl_branch\scripts\run_experiment.ps1' -Stage Run -RunId exp007_main_20261023 -Seeds 42,43,44 -Device cuda
```

该命令依次执行实验六冻结方法复评、C0及P1至P5的三个种子训练、验证集阈值校准、测试评估和多种子汇总。

## 本轮保存的记录

资格确认记录（包含更改前完整资格表）：D:\AIC\AIC-seek\experiments\exp007_psl_branch\outputs\exp007_main_20261023\baseline\annotation_eligibility_confirmation_20261024.json

机器可读标注核查：D:\AIC\AIC-seek\experiments\exp007_psl_branch\outputs\exp007_main_20261023\baseline\annotation_check_20261023.json

## 最后一次启动检查（2026-10-24）

实际运行 Stage Check 和 Stage Status，均以退出码0完成。CUDA可用，GPU为RTX 4060 Laptop，官方模型哈希匹配固定版本；train_ready=true、test_ready=true。正式训练尚未启动，可直接使用上面的Stage Run命令。
