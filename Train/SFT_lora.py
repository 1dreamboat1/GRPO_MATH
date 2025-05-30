import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset

# 设置环境变量以解决并行警告和显存优化
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置路径和参数
model_path = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
dataset_path = "/root/autodl-tmp/GRPO_MATH/gsm8k"
output_dir = "./lora_finetuned_qwen"
max_length = 256
batch_size = 4
gradient_accumulation_steps = 2
num_epochs = 3
lora_rank = 8
lora_alpha = 32
lora_dropout = 0.1
train_val_split_ratio = 0.8

# 数据预处理
def load_gsm8k_dataset(dataset_path):
    print(f"正在加载GSM8K数据集（仅训练集）...")
    main_train_path = f"{dataset_path}/main/train-00000-of-00001.parquet"
    
    # 加载Parquet格式训练集
    dataset = load_dataset("parquet", data_files={"train": main_train_path})
    train_data = list(dataset['train'])
    print(f"训练集大小: {len(train_data)}")
    
    # 构造输入输出对，保留CoT结构
    data = []
    for sample in train_data:
        input_text = f"问题: {sample['question']}\n回答: {sample['answer']}"
        data.append({"text": input_text, "answer": sample['answer']})
    
    # 转换为Dataset对象
    full_dataset = Dataset.from_list(data)
    
    # 划分训练集和验证集
    dataset_split = full_dataset.train_test_split(test_size=1 - train_val_split_ratio, seed=42)
    train_dataset = dataset_split['train']
    val_dataset = dataset_split['test']
    print(f"划分后训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    return train_dataset, val_dataset

# 加载数据集
train_dataset, val_dataset = load_gsm8k_dataset(dataset_path)

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

# 配置LoRA
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 数据集tokenization
def tokenize_function(examples):
    # 编码输入文本（问题+回答）
    encodings = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    # 编码答案部分作为labels（仅回答部分，包含CoT和####标记）
    answer_encodings = tokenizer(
        examples["answer"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    # 设置labels
    encodings["labels"] = answer_encodings["input_ids"].clone()
    # 将padding部分的labels设置为-100，忽略损失计算
    encodings["labels"][answer_encodings["attention_mask"] == 0] = -100
    return encodings

# 对训练集和验证集分别进行分词
train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=1)
val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=1)
train_dataset = train_dataset.remove_columns(["text", "answer"])
val_dataset = val_dataset.remove_columns(["text", "answer"])
train_dataset.set_format("torch")
val_dataset.set_format("torch")

# 训练配置
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False,
    bf16=True,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 清理显存
torch.cuda.empty_cache()

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# 推理测试
def generate_answer(question):
    input_text = f"问题: {question}\n回答: "
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取####标记的答案
    match = re.search(r'####\s*(\d+)', generated_text)
    final_answer = match.group(1) if match else "未找到答案"
    return {"generated_text": generated_text, "final_answer": final_answer}

# 测试样例
test_question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
result = generate_answer(test_question)
print(f"生成回答: {result['generated_text']}")
print(f"最终答案: {result['final_answer']}")