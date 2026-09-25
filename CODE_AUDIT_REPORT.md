# 代码审查报告

审查范围：`problem1_unaligned/`、`problem2_robust/`、`problem3_interpretability/`、主流程脚本及已生成结果文件。

审查方式：静态阅读、输入输出链路核对、结果文件命名核对。未修改模型代码、数据文件或已有实验结果。

## 结论摘要

当前项目的主流程基本完整，但不建议直接把现有结果作为严格可复现实验或论文最终证据。存在三个高优先级问题：

1. 解释图 `fig5_temporal_saliency_sample04.png` 使用人工构造曲线，不是模型实际输出。
2. 训练阶段把测试集逐 epoch 评估，并在测试集上比较 argmax 与阈值校准结果后取较优值，最终测试指标存在乐观偏差风险。
3. 注意力全缺失处理和时间池化没有严格屏蔽无效位置，边界输入的行为与代码注释不一致。

## 问题清单

### P0：解释图包含人工生成的模型证据

文件：`problem3_interpretability/generate_explanation_cards.py:64-75`

`steps`、`saliency` 由高斯函数和 `np.random.normal` 直接生成，并固定写入“Sample 04”“3.51s ~ 4.80s”“exercise, should I”等内容。该图不是由 `predict_att4_interpretable.py` 的 CSV 或模型反事实结果绘制。

影响：论文如果将该图表述为模型解释，会构成证据与实现不一致；重复运行还会产生不同曲线。

建议：从附件4预测 CSV 读取真实时间步、显著性和文本片段；如果没有保存这些字段，应先扩展预测脚本保存真实结果，再绘图。禁止使用随机曲线替代模型输出。

### P1：测试集被用于方法选择

文件：`problem2_robust/train.py:303-337`

训练后同时计算测试集 argmax 指标与阈值校准指标，然后执行：

```python
final_acc3 = max(test_ens['acc3'], cal_acc3)
final_f1_3 = max(test_ens['f1_3'], cal_f1_3)
```

这等价于根据测试集标签选择报告方案。阈值本身在验证集搜索是合理的，但“哪个方法最终报告”不能在测试集上二选一。

影响：测试集指标偏乐观，不能作为无偏最终评估。

建议：预先固定一种报告规则，例如始终报告验证集确定的 BUDT 结果；或在验证集上比较 argmax/BUDT 后锁定方案，再只在测试集上运行一次。

### P1：训练过程中反复查看测试集

文件：`problem2_robust/train.py:220-250` 附近的 `train_single_seed`

每个 epoch 都计算 `test_res = evaluate_loader(model, test_loader)` 并打印。虽然当前最佳 checkpoint 主要由验证分数选择，但持续查看测试集会让后续人工调参产生测试集反馈。

建议：训练阶段移除测试集评估；训练完成、模型和阈值完全锁定后只评估一次测试集。

### P1：全缺失注意力并未真正变成零信息

文件：`problem2_robust/model.py:46-63`

当 `valid_mask` 全为 0 时，所有 score 都被替换为相同的 `-1e4`，softmax 会得到近似均匀分布，而不是零输出。`nan_to_num` 只处理 NaN，不能修复这一逻辑。

影响：全缺失 key/value 时，query 仍会接收 value 的平均表示；输出行为与“缺失安全”注释不一致。

建议：对全缺失样本显式返回零 attention 输出，或在 softmax 前后使用有效性分支；同时为全缺失、部分缺失和完整输入增加单元测试。

### P1：时间池化包含无效/填充时间步

文件：`problem2_robust/model.py:191-194`

`sum_t = final_t.mean(dim=1)` 等语句对 50 个时间步直接平均，没有使用 `mask_t/mask_a/mask_v` 做有效长度归一化。

影响：不同有效长度的样本会受到不同程度的 padding/缺失表示影响；全缺失时甚至可能产生由 prompt 或注意力残差主导的平均特征。

建议：使用 `sum(mask * feature) / max(sum(mask), 1)`，并明确全缺失模态的 fallback 表示。

### P1：缺失 mask 的定义依赖特征数值，而非统一的有效长度定义

文件：`problem2_robust/dataset.py:73-80`、`problem2_robust/predict_att3.py:140-169`

文本、音频、视觉均通过 `abs(feature).sum(axis=-1) > 1e-5` 推断有效性，但文本在附件3中有时使用 BERT 的 attention mask，有时使用嵌入数值推断。两种路径的定义不完全等价。

建议：统一保存并传递显式 `mask_t/mask_a/mask_v`；文本优先使用原始 attention mask；音视频使用特征生成阶段的有效长度或缺失标记。

### P1：TTA 的循环移位会把边界时间步错误搬运到另一端

文件：`problem2_robust/predict_att3.py:46-73`

`torch.roll` 是循环移位，会把最后一个时间步放到第一个位置。对有明确开始/结束语义的文本、音频和视频，这不是“微偏”而是改变了序列边界。

建议：采用零填充平移，并同步更新 mask；或只对连续有效区间做局部扰动，并对 TTA 结果进行单独消融验证。

### P1：报告的 seed 数量与实际配置不一致

文件：`problem2_robust/train.py:27`、`train.py:372`、`run_local_pipeline.py`

`SEEDS = [42, 123, 777]` 实际是 3 个 seed，但日志和 checkpoint 注释多处写成“5-Seed Ensemble”。

影响：论文、日志和 checkpoint 元数据可能产生事实不一致。

建议：统一改成实际数量，或明确补齐 seed 并重新训练；不要仅修改文字掩盖实验数量差异。

### P2：解释贡献的表述强于实际方法

文件：`problem3_interpretability/predict_att4_interpretable.py:130-186`

当前 `delta_t/delta_a/delta_v` 是遮挡一个模态后回归输出变化的敏感性分数。它可以称为“遮挡敏感性”或“反事实扰动贡献”，不能直接等价为严格因果贡献。

建议：在论文和 CSV 字段中使用“遮挡敏感性/相对贡献”，并说明模型重算、非独立贡献和模态交互带来的限制。

### P2：文件路径和异常处理影响可复现性

多个脚本固定使用 `X:\mathModelHands`、`/root/E_Problem`，并存在裸 `except` 或宽泛 `except Exception`。

建议：统一从项目根目录或命令行参数解析路径；异常至少记录文件名、异常类型和上下文，数据校验失败时应终止而不是静默跳过。

## 数据流核对

```text
DATA/Att2-Extracted-Features/aligned_50.pkl
        |
        v
MOSEIDataset
  |-- train: 训练统计量 + 随机局部缺失 + teacher 完整输入
  |-- valid: 训练统计量 + 不做随机缺失
  `-- test : 训练统计量 + 不做随机缺失
        |
        v
RobustMultimodalModel
  投影 -> Bi-GRU -> 跨模态注意力 -> 门控融合 -> 分类/回归
        |
        +--> 验证集阈值搜索
        +--> 测试集最终指标
        +--> 附件3 TTA 预测
        `--> 附件4 遮挡敏感性解释
```

训练集标准化统计量传给验证集和测试集，这一点是正确方向；但最终指标的方案选择仍需要修正。

## 结果文件一致性核对

- `problem2_robust/problem2_key_metrics.json/csv`：应在修正测试集方案锁定流程后重新生成。
- `problem2_robust/附件3_*.csv`：两个输出文件当前由同一 DataFrame 写入，内容重复，应明确一个是局部缺失、一个是多模态缺失，或重命名避免语义误导。
- `problem3_interpretability/附件4_可解释专项测试集预测与解释结果.csv`：可作为模型解释输入，但当前图5没有使用它生成。
- 论文 LaTeX 中的指标需要与重新锁定后的 JSON/CSV 逐项比对，不能只看文件名或旧日志。

## PDF 核对状态

本次环境中的 `pdfinfo`、`pdftotext` 和 `pdftoppm` 实际解析器指向 MiKTeX，并因无法写入 MiKTeX 配置目录而失败；浏览器本地 PDF 标签也被安全策略阻止。因此本报告没有把 `problem.pdf` 的具体题目文字伪装成已核对事实。

已核对的是代码自身、数据目录命名、附件1-4相关入口和论文源文件中的问题章节。要完成逐条题目映射，需要提供可解析的 PDF 文本/截图，或配置可写的 Poppler/PDF 运行时后重新执行该部分。

## 建议修复顺序

1. 删除人工解释曲线，改为真实模型输出。
2. 锁定验证集方案并移除训练期测试集评估。
3. 修复全缺失注意力和 mask-aware pooling。
4. 统一显式 mask 定义，修正 TTA 边界处理。
5. 统一 seed 数量、路径解析和异常策略。
6. 重新运行全流程，再同步论文数字和题目映射表。
