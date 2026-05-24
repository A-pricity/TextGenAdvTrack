# Codex Data Generation Playbook

在没有 `OpenAI API key` 的情况下，当前项目改为使用对话式 `Codex` 直接生成训练原料。

## 原则

- `human`：应尽量来自真实公开文本，不建议完全用模型伪造
- `ai_original`：可直接由 Codex 按 prompt registry 生成
- `ai_rewritten`：可直接由 Codex 对 `ai_original` 做逃逸式改写

## 推荐做法

### 1. 先用仓库内种子数据跑通

当前目录下已放入一批小规模种子文件：

- `datasets/training/human.csv`
- `datasets/training/ai_original.csv`
- `datasets/training/ai_rewritten.csv`

作用：

- 跑通 split builder
- 跑通 detection baseline
- 跑通 evasion 闭环

### 2. 再按批次扩充

每次扩充建议按语言和任务类型分批：

- 中文 `qa/fact/tutorial/daily`
- 英文 `qa/fact/tutorial/daily`
- 俄文 `qa/fact/tutorial/daily`

每批建议只扩 `10~30` 条，方便人工抽查质量。

## 给 Codex 的生成提示模板

### 生成 `ai_original`

```text
请生成 N 条 AI 原始文本训练样本，输出为纯 CSV 行，不要解释。
要求：
1. language={zh|en|ru}
2. domain={qa|fact_explanation|opinion|tutorial|daily_writing}
3. label 固定为 0
4. text_type 固定为 ai_original
5. source_name 固定为 self_generated
6. source_model 固定填你指定的模型名占位符
7. prompt_type 和 prompt_id 要与任务一致
8. text 要自然、完整、像真实模型回答
9. 不要输出表头，只输出数据行
字段顺序：
sample_id,language,domain,label,text_type,source_name,source_model,prompt_type,prompt_id,rewrite_type,parent_id,text
```

### 生成 `ai_rewritten`

```text
请把以下 ai_original 样本改写为更像人类写作的文本，输出为纯 CSV 行，不要解释。
要求：
1. 保留原意
2. 减少明显模板化表达
3. 可使用 paraphrase / reorder / compress / expand / synonym
4. label 固定为 0
5. text_type 固定为 ai_rewritten
6. parent_id 必须指向原始 ai_original 的 sample_id
7. 不要输出表头，只输出数据行
字段顺序：
sample_id,language,domain,label,text_type,source_name,source_model,prompt_type,prompt_id,rewrite_type,parent_id,text
```

## 质量检查

扩充后至少检查：

1. `sample_id` 是否唯一
2. `language` / `domain` / `prompt_type` 是否匹配
3. `ai_rewritten` 是否都带 `parent_id`
4. 文本是否过短、过像模板、或语义跑偏

## 下一步

1. 先用现有种子文件构建 split
2. 再分批让 Codex 扩充 `ai_original`
3. 再分批扩充 `ai_rewritten`
4. human 部分优先从真实公开文本补充，而不是完全用模型代替
