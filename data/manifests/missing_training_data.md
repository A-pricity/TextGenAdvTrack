# Missing Training Data Checklist

官方基础数据已经下载完成，但训练主模型仍缺少合规训练数据。

## 已具备

- 官方验证集：`datasets/official/raw/UCAS_AISAD_TEXT-val.csv`
- 官方验证标签：`datasets/official/raw/UCAS_AISAD_TEXT-val_label.csv`
- 官方测试集 1：`datasets/official/raw/UCAS_AISAD_TEXT-test1.csv`

这些文件可用于：

- detection 本地评测
- detection 提交文件导出
- evasion 种子样本抽取

## 仍缺失

### 1. Detection 主训练集

需要补齐：

- `human`
- `AI-original`
- `AI-rewritten`

约束：

- 不得使用 `M4 / HC3`
- 需要中英俄覆盖
- 需要多域、多生成器

### 2. Detection 本地开发集

需要补齐：

- `det-train.csv`
- `det-dev.csv`
- `det-rewrite-dev.csv`

这些文件需符合项目 schema：

```text
sample_id,split,language,domain,label,text_type,source_name,source_model,prompt_type,prompt_id,rewrite_type,parent_id,text
```

### 3. Evasion 候选生成源集

需要补齐或扩展：

- `eva-source.csv`

当前可从 `official/val_machine_only.csv` 启动一个最小版本，但最终仍建议加入：

- 自生成 `AI-original`
- 多模型来源
- 多语言覆盖

### 4. Evasion 候选与筛选集

运行当前 pipeline 前仍需准备：

- `eva-candidates.csv`
- `eva-selected.csv`

当前代码已支持自动生成和筛选，但仍需要合适的输入源。

## 推荐下一步

1. 运行 `prepare-official-data`，把下载好的 raw 数据整理进工程目录
2. 基于 `val_machine_only.csv` 跑通 evasion 最小闭环
3. 准备合规的 `human / AI-original / AI-rewritten` 训练数据
4. 再切换到真正的 detection 主模型训练
