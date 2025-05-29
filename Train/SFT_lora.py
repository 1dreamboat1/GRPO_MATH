import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 1. 加载模型和分词器
model_name = "Qwen/Qwen-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

# 2. 加载GSM8K数据集
dataset = load_dataset("gsm8k", "main")
train_dataset = dataset["train"]

# 3. 数据预处理
def preprocess_data(examples):
    inputs = []
    outputs = []
    for question, answer in zip(examples["question"], examples["answer"]):
        # 构造输入提示
        input_text = f"问题: {question}\n回答: "
        # 构造目标输出（包含CoT结构）
        target_text = f"{answer}"
        inputs.append(input_text)
        outputs.append(target_text)
    
    # 分词
    encodings = tokenizer(inputs, truncation=True, padding=True, max_length=512, return_tensors="pt")
    encodings["labels"] = tokenizer(outputs, truncation=True, padding=True, max_length=512, return_tensors="pt")["input_ids"]
    return encodings

# 应用预处理
train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)

# 4. 配置LoRA
lora_config = LoraConfig(
    r=8,  # LoRA的秩
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 针对Qwen模型的注意力层
    lora_dropout=0.1,  # Dropout率
    bias="none",
    task_type="CAUSAL_LM"
)

# 将LoRA应用到模型
model = get_peft_model(model, lora_config)

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./qwen_lora_output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,  # 混合精度训练
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",  # 禁用wandb等外部日志
)

# 6. 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# 7. 开始训练
trainer.train()

# 8. 保存微调后的模型和分词器
output_dir = "./qwen_lora_output/final_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"模型和分词器已保存至 {output_dir}")