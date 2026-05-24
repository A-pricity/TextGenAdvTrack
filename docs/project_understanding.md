# Project Understanding

## 1. 这个项目现在是什么状态

`/home/apricity/workspace/repos/TextGenAdvTrack` 当前是新的整理目录。

本地真正有内容的是：

- `/home/apricity/workspace/projects/TextGenAdvTrack`

它已经有一版可运行底稿，不是只有说明文档。

## 2. 本地底稿已经具备什么

我确认到这些部分已经存在：

- `official/`
  - 官方赛题说明副本
- `research/`
  - 方法调研和证据分析
- `scripts/`
  - 统一执行入口
  - smoke test
  - detection / ensemble / eval 脚本
- `src/training/`
  - 训练脚本
  - 推理脚本
- `src/models/`
  - `binoculars_detector.py`
  - `fast_detect_gpt_detector.py`
  - `ensemble_detector.py`
- `src/evaluation/`
  - 指标计算逻辑

## 3. 官方任务接口是什么

从本地副本 `official/AI_Text Detection/README.md` 看，官方接口要求非常清楚：

### 输入

CSV 文件，至少包含：

- `prompt`
- `text`

验证标签 CSV：

- `label`

其中：

- `0` 表示 `machine_text`
- `1` 表示 `human_text`

### 输出

需要提交一个 Excel 文件 `<team-name>.xlsx`，包含两个 sheet：

1. `predictions`
   - `prompt`
   - `text_prediction`
2. `time`
   - `Data Volume`
   - `Time`

关键点：

- `text_prediction` 必须表示“越大越像 human”
- 不能改列名
- 不能改样本顺序

## 4. 评测目标是什么

官方底稿给出的分数为：

```text
Final_Score = (0.6 × AUC + 0.3 × ACC + 0.1 × F1) / 100
```

这意味着：

- `AUC` 最重要
- 不能只追求某个固定阈值下的准确率
- 输出分数的排序质量和校准能力很重要

## 5. 现有代码路线是什么

从 `scripts/run_pipeline.py` 和 `src/models/` 可见，底稿已经不是单模型思路，而是三层能力：

### 5.1 主流程

- `smoke`
- `binoculars`
- `fdgpt`
- `ensemble`
- `eval`
- `train`
- `predict`

说明项目已经把：

- 零样本检测
- 融合检测
- 微调训练
- 本地评测

都作为独立子流程准备好了。

### 5.2 训练能力

`src/training/train_detector.py` 当前默认训练的是：

- `xlm-roberta-base`

并且支持：

- 从多个训练数据目录合并数据
- 自动推断文本列和标签列
- 统一映射标签到 `machine=0 / human=1`
- 直接算 `accuracy / f1 / auc`

这说明底稿已经偏向：

- 多源训练数据
- 多语言检测
- 监督学习主线

### 5.3 融合能力

`src/models/ensemble_detector.py` 说明现有思路是：

- 一个 `Binoculars` 分支
- 一个 `Fast-DetectGPT` 分支
- 对两个分数做加权融合

这更像“零样本辅助模块”，而不是最终唯一主模型。

## 6. 技术路线应该怎么选

根据本地 `research/evidence_analysis.md` 的结论，这个赛题更适合：

- `Fine-tuned 模型为主`
- `Zero-shot 方法为辅`

原因很直接：

1. 官方接口天然适合分类器输出
2. AUC 权重大，要求分数稳定
3. 纯零样本方案跨域表现通常更差
4. 测试集 2 带逃逸攻击，单一路线风险大

## 7. 现阶段最缺的不是代码，而是这几件事

1. 明确哪些训练数据可以合法使用
2. 决定第一版 baseline
3. 跑通验证集完整提交流程
4. 补实验记录和图表
5. 准备最终 PPT 叙事

## 8. 结论

如果只看当前阶段，最合理的理解是：

- 比赛接口已经明确
- 本地底稿已经有一套可复用实现
- 你们真正要补的是“合规数据 + baseline + 实验结果 + 展示材料”

不是先去重写整套框架。
