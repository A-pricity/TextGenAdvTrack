# Dual-Track Baseline Design

## Overview

本 spec 定义国科大课程实践 `生成文本对抗赛道` 的双子赛道 baseline 方案。

范围明确为：

1. `生成文本检测赛道`
2. `逃逸检测赛道`

两条子赛道都必须完成，且采用 `分别设计、分别优化、有限联动` 的组织方式。

本 spec 取代此前只围绕单一检测赛道的设计思路。

## Official Ground Truth

以官方仓库为准：

- 仓库：`https://github.com/UCASAISecAdv/TextGenAdvTrack-2026Spring`
- 检测赛道：`AI_Text Detection`
- 逃逸赛道：`AI_Text Evasion`

### Detection scoring

检测赛道指标：

```text
Final_Score = (0.6 * AUC + 0.3 * ACC + 0.1 * F1) / 100
```

结论：

- `AUC` 是第一优先级
- 模型必须输出稳定的 human probability
- 训练是必要的，不是可选项

### Evasion scoring

逃逸赛道指标：

- 使用 `3` 个不同模型计算 `1-AUC`
- 再取平均

结论：

- 不能只针对单一本地检测器过拟合
- 逃逸策略必须兼顾通用性和本地可优化性

## High-Level Strategy

### Organization choice

用户已确认：

- 两条子赛道 `分别设计`
- 两条子赛道 `均衡投入`

### Shared principle

虽然两条子赛道分开优化，但允许一个有限联动：

- 检测赛道训练出的本地检测器
- 可作为逃逸赛道的代理评分器之一

### Core choice for evasion

逃逸赛道采用：

```text
本地检测器指导 + 通用改写约束
```

即：

- 用本地代理检测器提供优化信号
- 但改写策略本身尽量通用，不绑定单一 detector

## Dual-Track Baseline

## 1. Detection track baseline

### Goal

训练一个能够在官方检测验证/测试流程中提交 `.xlsx` 的主检测器。

### Backbone shortlist

第一阶段只比较两个 backbone：

1. `DeBERTa-v3-base`
2. `XLM-R base`

理由：

- 单卡 `4090 24GB` 可训练
- 一个偏通用判别能力
- 一个偏跨语言稳健性

### Training requirement

检测赛道必须训练。

理由：

- 官方允许自选训练集，但禁止 `M4 / HC3`
- 这说明官方默认参赛者自建训练数据并训练自己的 detector

### Detection baseline pipeline

```text
公开 human 数据 + 自生成 AI 数据
-> 构造 det-train / det-dev
-> 训练 backbone A/B
-> official-val 上比较 AUC/ACC/F1
-> 选主模型
-> 导出官方 .xlsx
```

## 2. Evasion track baseline

### Goal

对 machine text 进行人类化改写，生成官方要求的提交 `.csv`，并尽量降低官方隐藏检测模型的识别能力。

### Evasion baseline pipeline

```text
AI-original
-> 分层抽样形成 eva-source
-> 多策略生成改写候选
-> 本地代理检测器评分
-> 多代理筛选
-> 输出最终 eva csv
```

### Why not pure black-box local overfitting

官方逃逸赛道按 `3` 个不同模型平均 `1-AUC` 评测。

因此：

- 纯粹只攻击一个本地 detector，风险太高
- 纯通用改写又缺少足够强的优化信号

所以最合理的 baseline 是：

- `本地 detector 提供反馈`
- `改写策略保持通用`

## Data Strategy

用户已确认的数据大方向：

- `公开数据集打底`
- `自己额外生成 AI 文本`
- `中英俄都覆盖`
- `多种模型，中等 prompt 覆盖`
- `AI-rewritten 少量进入检测赛道训练`
- 数据文件以 `csv` 为主

## Data Splits

## 1. Detection training set: `det-train`

组成：

1. `human`
2. `AI-original`
3. 少量 `AI-rewritten`

作用：

- 训练检测主模型

建议比例起点：

```text
human : AI-original : AI-rewritten = 1 : 1 : 0.2~0.4
```

## 2. Detection dev set: `det-dev`

组成：

1. `human`
2. `AI-original`

作用：

- backbone 选择
- 超参选择
- 主看原始分布上的 `AUC`

## 3. Detection rewrite robustness dev: `det-rewrite-dev`

组成：

1. `human`
2. `AI-rewritten`

作用：

- 检验 rewritten 混入训练是否真的提升鲁棒性

硬约束：

- `det-rewrite-dev` 不能和训练增强使用完全同源的 rewrite 模板和改写器

## 4. Evasion source set: `eva-source`

组成：

- 按语言、模型、任务类型分层抽样的 `AI-original`

作用：

- 作为逃逸攻击输入源

## 5. Evasion candidate set: `eva-candidates`

组成：

- 对 `eva-source` 的每条样本生成多个改写候选

作用：

- 做本地代理筛选
- 分析哪类攻击更有效

## 6. Evasion selected set: `eva-selected`

组成：

- 每条源文本最终保留的最优候选

作用：

- 形成最终逃逸赛道提交候选

## Small-Scale Validation Version

用户已确认当前阶段先做：

```text
小规模但完整可跑通的验证版
```

目标不是最终高分，而是先验证双赛道 pipeline 闭环。

### Detection data sizes

#### `det-train`

- `human = 3000`
- `AI-original = 3000`
- `AI-rewritten = 1200`
- 总量 `7200`

#### `det-dev`

- `human = 500`
- `AI-original = 500`
- 总量 `1000`

#### `det-rewrite-dev`

- `human = 300`
- `AI-rewritten = 300`
- 总量 `600`

### Evasion data sizes

#### `eva-source`

- 中文 `100`
- 英文 `100`
- 俄文 `100`
- 总量 `300`

#### `eva-candidates`

每条源文本生成 `6` 个候选：

1. 转述 `2`
2. 句式重排 `1`
3. 压缩 `1`
4. 扩写 `1`
5. 同义/词汇替换 `1`

总量：

```text
300 * 6 = 1800
```

## Language Mix

用户已确认语言结构：

- 中文 `40%`
- 英文 `35%`
- 俄文 `25%`

### Detection split allocation

#### `det-train`

| 类别 | 中文 | 英文 | 俄文 | 小计 |
|---|---:|---:|---:|---:|
| human | 1200 | 1050 | 750 | 3000 |
| AI-original | 1200 | 1050 | 750 | 3000 |
| AI-rewritten | 480 | 420 | 300 | 1200 |

#### `det-dev`

| 类别 | 中文 | 英文 | 俄文 | 小计 |
|---|---:|---:|---:|---:|
| human | 200 | 175 | 125 | 500 |
| AI-original | 200 | 175 | 125 | 500 |

#### `det-rewrite-dev`

| 类别 | 中文 | 英文 | 俄文 | 小计 |
|---|---:|---:|---:|---:|
| human | 120 | 105 | 75 | 300 |
| AI-rewritten | 120 | 105 | 75 | 300 |

## AI-Original Generation Plan

用户已确认生成来源为：

```text
闭源 API + 开源模型都用
```

### Generator pool

第一阶段使用：

- 闭源模型 `2` 个
- 开源模型 `2` 个

记为：

- `Model-C1`
- `Model-C2`
- `Model-O1`
- `Model-O2`

### Allocation

`AI-original = 3000`

| 模型 | 配额 | 占比 |
|---|---:|---:|
| Model-C1 | 900 | 30% |
| Model-C2 | 600 | 20% |
| Model-O1 | 900 | 30% |
| Model-O2 | 600 | 20% |

### Rationale

- 闭源模型负责高质量、强迷惑性文本
- 开源模型负责风格差异与家族覆盖
- 避免训练分布被单一模型支配

## Prompt Plan

prompt 采用 `中等覆盖`，不追求极大规模。

### Prompt categories

1. 事实解释
2. 问题回答
3. 摘要改写
4. 观点表达
5. 教程说明
6. 日常写作

### Allocation

`AI-original = 3000`

每类约 `500`

按语言分配：

- 中文：每类约 `200`
- 英文：每类约 `175`
- 俄文：每类约 `125`

## AI-Rewritten Plan

### Role

`AI-rewritten` 同时服务两个目标：

1. 少量进入检测赛道训练，增强鲁棒性
2. 作为逃逸赛道改写能力的基础样本

### Detection-side role

用户已确认：

```text
AI-rewritten 少量进入检测赛道训练
```

这意味着：

- 不把 rewritten 当成主分布
- 只做有限鲁棒增强

### Rewrite strategy mix

第一阶段主改写策略：

1. 转述
2. 句式重排
3. 压缩
4. 扩写

#### `AI-rewritten = 1200`

每类约 `300`

| 改写类型 | 中文 | 英文 | 俄文 | 小计 |
|---|---:|---:|---:|---:|
| 转述 | 120 | 105 | 75 | 300 |
| 句式重排 | 120 | 105 | 75 | 300 |
| 压缩 | 120 | 105 | 75 | 300 |
| 扩写 | 120 | 105 | 75 | 300 |

### Rewrite generation source mix

- 闭源模型改写：`50%`
- 开源模型改写：`30%`
- 轻量规则扰动：`20%`

## Proxy Evaluation for Evasion

第一轮代理评测使用 `2` 个代理：

1. 检测赛道最佳主模型
2. 第二 backbone 或轻量统计分支

### Selection rule

候选文本优先保留：

- 在两个代理上都更像 human
- 且语义保持较好

如果只在单一代理上显著提升，而另一代理恶化明显，则默认丢弃。

## CSV-First Data Schema

用户已确认：

```text
数据以 csv 为主
```

因此必须一开始就把 metadata 字段定义完整。

## Directory structure

```text
data/
├── detection/
│   ├── train/
│   │   ├── det_train.csv
│   │   ├── det_dev.csv
│   │   └── det_rewrite_dev.csv
│   └── official/
│       ├── val.csv
│       └── val_label.csv
├── evasion/
│   ├── source/
│   │   └── eva_source.csv
│   ├── candidates/
│   │   └── eva_candidates.csv
│   ├── selected/
│   │   └── eva_selected.csv
│   └── official/
│       └── test_input.csv
└── manifests/
    ├── data_sources.csv
    ├── model_registry.csv
    └── prompt_registry.csv
```

## Detection CSV schema

适用于：

- `det_train.csv`
- `det_dev.csv`
- `det_rewrite_dev.csv`

字段：

```csv
sample_id,split,language,domain,label,text_type,source_name,source_model,prompt_type,prompt_id,rewrite_type,parent_id,text
```

字段说明：

- `sample_id`：唯一 id
- `split`：`train/dev/rewrite_dev`
- `language`：`zh/en/ru`
- `domain`：文本域
- `label`：`1=human, 0=machine`
- `text_type`：`human / ai_original / ai_rewritten`
- `source_name`：数据来源名
- `source_model`：AI 文本的生成模型
- `prompt_type`：prompt 类别
- `prompt_id`：prompt 编号
- `rewrite_type`：改写类型
- `parent_id`：改写前原文 id
- `text`：正文

## Evasion source CSV schema

字段：

```csv
sample_id,language,domain,source_model,prompt_type,prompt_id,source_text
```

## Evasion candidates CSV schema

字段：

```csv
candidate_id,parent_id,language,domain,source_model,prompt_type,rewrite_model,rewrite_type,semantic_score,proxy_score_1,proxy_score_2,selected,text
```

## Evasion selected CSV schema

字段：

```csv
sample_id,parent_id,language,final_text,proxy_score_1,proxy_score_2,selection_reason
```

## Manifest schemas

### `data_sources.csv`

```csv
source_name,source_type,language,license,used_for,notes
```

### `model_registry.csv`

```csv
model_name,model_family,model_role,open_or_closed,languages,used_for,notes
```

### `prompt_registry.csv`

```csv
prompt_id,language,prompt_type,prompt_text,used_models,notes
```

## Acceptance Criteria

第一阶段验证版完成时，至少要满足：

1. 检测赛道能训练出一个可提交 `.xlsx` 的模型
2. 检测赛道能比较两个 backbone
3. 能验证少量 `AI-rewritten` 混入是否值得
4. 逃逸赛道能从 `eva-source -> candidates -> selected` 跑通完整闭环
5. 能比较至少两种改写策略的有效性
6. 所有数据均可通过 csv metadata 追溯来源、模型、语言和改写类型

## Risks and Controls

### Risk 1: Detection overfits rewritten distribution

控制：

- `AI-rewritten` 只小比例进入 `det-train`
- 用 `det-dev` + `det-rewrite-dev` 双验证

### Risk 2: Evasion overfits one local detector

控制：

- 至少两代理评分
- 改写策略保持通用

### Risk 3: CSV metadata not enough

控制：

- 所有关键 lineage 字段前置设计
- 使用 `manifests` 维护来源登记

### Risk 4: Data engineering becomes too large

控制：

- 当前只做小规模验证版
- 先跑通闭环，再按比例放大

## Deliverables

第一阶段应产出：

1. 双赛道数据目录与 csv 文件
2. 数据来源登记表
3. 模型登记表
4. prompt 登记表
5. 检测赛道 baseline 结果
6. 逃逸赛道候选筛选结果
7. 双赛道初步 ablation 结论

## Obsoleted Design

以下旧设计不再作为主规范：

- `2026-05-11-high-score-strategy-design.md`

原因：

- 它没有完整覆盖双子赛道
- 它没有吸收官方逃逸评测方式这一关键约束
