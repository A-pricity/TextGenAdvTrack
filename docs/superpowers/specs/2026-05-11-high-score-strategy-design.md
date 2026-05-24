# High-Score Strategy Design

## Overview

目标是在单卡 `RTX 4090 24GB` 条件下，为课程实践中的 `生成文本对抗赛道` 设计一条高分、可复现、可答辩的实施路线。

约束如下：

- 不下载官方仓库作为前置条件
- 训练阶段禁止使用 `M4` 和 `HC3`
- 最终既要争取榜单效果分，也要保留可讲清楚的新颖性
- 当前优先目标是 `生成文本检测赛道`

本设计不直接进入代码实现，而是先定义：

1. 最小可行高分路线
2. 训练数据构造原则
3. 验证协议
4. 组件进入门槛
5. 风险控制

## Agreed Strategy

经过显式 debate，最终一致方案不是“同时堆叠所有高级组件”，而是受控增益链：

```text
强监督主模型
-> 轻量逃逸鲁棒增强
-> 概率校准 / 轻量融合
-> 可选蒸馏
```

### Why this strategy

- `AUC` 权重最高，主模型必须能输出稳定分数
- 测试集包含逃逸攻击，不能只对原始 AI 文本有效
- 单卡 `4090` 的主要瓶颈是实验吞吐，不是显存
- 课程赛需要“有新意但不失控”的路线

## Scope

### In scope

- 训练一个主检测模型
- 对比至少两个 backbone
- 构造合法训练数据
- 构造本地鲁棒验证集 `rewrite-dev`
- 做轻量逃逸增强
- 做概率校准
- 在有明确互补时做 late fusion

### Out of scope for phase 1

- 复杂蒸馏
- 多任务联合训练
- 大规模 stacking
- 重型生成式教师系统

## Backbone Strategy

第一阶段至少比较以下主模型候选：

1. `DeBERTa-v3-base`
2. `XLM-R base`

如果数据盘点表明中文占绝对主导，再增加一个中文 backbone 候选，但默认不超过 `3` 个 backbone。

### Selection rule

主模型不是按主观偏好决定，而是按验证结果决定：

- 原始验证集 `AUC`
- 原始验证集 `ACC`
- 原始验证集 `F1`
- `rewrite-dev` 上的鲁棒表现
- 推理时间和训练成本

## Data Design

这是整个项目的第一优先级。

### Training split types

训练数据需要显式分成 3 类：

1. `human`
2. `AI-original`
3. `AI-rewritten`

### Data principles

- 不得使用 `M4 / HC3` 做训练
- human 文本要尽量覆盖多域，而不是只来自单一来源
- AI 文本要覆盖多个生成器或生成风格
- rewrite 不能只靠一种模板或一种模型
- 避免 topic leakage 和格式泄漏

### Rewrite augmentation principles

允许的轻量逃逸增强包括：

- 转述
- 同义替换
- 句式重排
- 压缩
- 扩写

不要求一开始做大规模复杂攻击库，优先保证多样性和可控性。

## Validation Protocol

验证必须同时看“原始效果”和“逃逸鲁棒性”，否则会误判增强是否有效。

### Validation sets

1. `official-val`
   作用：衡量主榜单相关指标

2. `rewrite-dev`
   作用：衡量对逃逸重写的鲁棒性

### Hard constraint

`rewrite-dev` 必须和训练增强解耦：

- 不能用同一套改写模板同时生成训练增强和 `rewrite-dev`
- 否则鲁棒增益可能只是模板记忆

### Acceptance rule for robustness augmentation

只有同时满足以下条件，才接受某个增强配置：

1. `official-val` 上 `AUC` 没有明显下降
2. `rewrite-dev` 上鲁棒指标有明确提升

如果鲁棒性提升但原始 `AUC` 明显下降，则该增强配置不进入主线。

## Training Schedule

### Phase 1: Strong supervised baseline

- 使用 `human` 与 `AI-original` 训练基础二分类器
- 建立 backbone 对比基线
- 输出第一版主模型

### Phase 2: Curriculum robustness tuning

- 在第一阶段主模型上 continued fine-tuning
- 小比例混入 `AI-rewritten`
- 控制增强比例，避免边界变钝

### Phase 3: Calibration

- 对输出分数做概率校准
- 目标是提升 `AUC` 和最终提交分数稳定性

### Phase 4: Late fusion

只在辅助分支体现稳定互补时引入。

辅助分支可以是：

- 轻量统计特征分支
- 第二个 backbone

### Gate for late fusion

只有当辅助分支：

1. 单独有可解释信号
2. 与主模型存在稳定互补
3. 校准后确实提升验证结果

才保留 fusion。否则默认单模型。

## Distillation Policy

蒸馏不进入第一实施方案。

只有以下条件同时满足时，才进入后续阶段：

1. 异构 ensemble 明显强于单模型
2. ensemble 推理成本成为真实问题
3. 教师输出稳定可靠
4. student 蒸馏目标被清晰定义

唯一接受的蒸馏形式：

- teacher：已验证有效的异构 ensemble
- student：单模型 detector
- target：soft probability + hard label joint distillation

如果这些条件不满足，则不做蒸馏。

## Error Handling

### Risk 1: Data distribution mismatch

处理方式：

- 优先扩展域覆盖和生成器覆盖
- 不急于上复杂模型技巧

### Risk 2: Robustness hurts AUC

处理方式：

- curriculum mixing
- 小比例增强
- 双验证协议约束

### Risk 3: Over-design under 4090 budget

处理方式：

- backbone 数量上限控制
- 蒸馏后置
- fusion 设门槛

### Risk 4: Novelty without score gain

处理方式：

- 先锁 baseline 和效果分
- 再把鲁棒增强包装为新颖性主线

## Testing Plan

实验报告至少要包含以下 ablation：

1. 不同 backbone 对比
2. 是否加入 `AI-rewritten`
3. 不同增强比例对比
4. 是否做校准
5. 是否做 fusion

最少要能回答：

- 主模型本身能到什么水平
- 逃逸增强是否真的带来收益
- fusion 是否有真实互补

## Deliverables

第一阶段完成后应产出：

1. 训练数据来源清单
2. 数据划分与合规说明
3. backbone 对比结果
4. baseline 模型
5. `official-val` 与 `rewrite-dev` 评测报告

## Decision Summary

最终一致结论如下：

- 需要训练主模型
- 第一优先级是数据分布工程
- 第一实施方案不包含蒸馏
- 鲁棒增强必须受验证协议约束
- fusion 必须有验证增益门槛
- 整体路线应当是“受控增益链”，而不是堆组件

## Open Questions

在进入实现计划前，仍需明确：

1. 可用训练数据源有哪些
2. 数据语言比例是什么
3. 是否优先冲检测赛道单榜，还是同时兼顾逃逸赛道叙事
4. 本地是否已有可直接使用的数据下载结果
