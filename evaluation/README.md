# 模型批量测试工具

此工具用于批量测试在GSM8K数据集上微调的模型，包括LoRA和GRPO微调模型。

## 功能特点

- 支持批量测试不同类型的微调模型 (LoRA, GRPO)
- 评估模型在数学推理任务上的表现
- 分析模型生成的答案正确率和格式正确率
- 生成详细的评估报告和统计结果

## 使用方法

### 基本用法

```bash
python batch_test.py --model ./lora_finetuned_qwen --sample_size 100
```

### 参数说明

- `--base_model`: 基础模型路径，默认为 `/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B`
- `--model`: 微调模型路径，必需参数
- `--dataset`: GSM8K数据集路径，默认为 `/root/autodl-tmp/GRPO_MATH/gsm8k`
- `--output`: 评估结果输出目录，默认为 `./evaluation_results`
- `--model_type`: 模型类型，可选 `lora` 或 `grpo`，默认为 `lora`
- `--sample_size`: 测试样本数量，默认为 100
- `--batch_size`: 批处理大小，默认为 8
- `--temperature`: 生成温度，默认为 0.1
- `--max_new_tokens`: 最大生成token数量，默认为 256
- `--no_bf16`: 不使用bf16精度，默认使用bf16

### 测试LoRA模型示例

```bash
python batch_test.py --model ./lora_finetuned_qwen --model_type lora --sample_size 200 --batch_size 16
```

### 测试GRPO模型示例

```bash
python batch_test.py --model ./grpo_finetuned_qwen --model_type grpo --sample_size 200 --batch_size 16
```

## 输出结果

测试结果将保存在指定的输出目录中，文件名格式为：`{model_name}_{timestamp}_test_results.json`。
结果包含以下信息：

- 测试样本数量
- 答案正确数量和准确率
- 格式正确数量和准确率
- 每个样本的详细信息（问题、答案、预测结果等）

## 注意事项

- 确保有足够的显存进行批处理，否则可以降低 `batch_size`
- 对于较大的模型，建议增加 `max_new_tokens` 以允许更复杂的推理过程
