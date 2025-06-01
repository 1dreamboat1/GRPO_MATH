from datasets import load_dataset,Dataset
import pandas as pd
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


dataset_path: str = "/root/autodl-tmp/GRPO_MATH/gsm8k"
main_train_path = f"{dataset_path}/main/train-00000-of-00001.parquet"
main_test_path = f"{dataset_path}/main/test-00000-of-00001.parquet"
dataset = load_dataset("parquet", data_files={
    "train": main_train_path,
    "test": main_test_path
})
print(dataset)
model_path: str = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"


# 加载分词器
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')?
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    padding_side='right')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
        
# 加载模型
original_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 
)
        
logger.info(f"模型参数量: {original_model.num_parameters():,}")

#将推理数据集转换为对话格式
def generate_conversation(examples) :
    question = examples ["question"]
    answer = examples ["answer"]
    conversations = []
    for question, answer in zip(question, answer):
        conversations.append( [
            {"role" : "user","content" : question},
            {"role" : "assistant", "content" : answer},
        ])
    return { "conversations": conversations,}

#将转换后的推理数据集应用对话模板
mapped_dataset = dataset.map(generate_conversation, batched=True)
reasoning_conversations = tokenizer.apply_chat_template(
    mapped_dataset["train"]["conversations"],  # 注意这里是 "train" 分割
    tokenize=False,  # 不进行分词,仅应用模板
)


# print(reasoning_conversations)
print(reasoning_conversations[0])

# 将对话数据转换为 Hugging Face Dataset 格式
data = pd.concat([
    pd.Series(reasoning_conversations),#推理对话数据
])
data.name ="text"# 设置数据列名为"text"
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
print(combined_dataset)


#微调
#使用HuggingFace TRL的SFTTrainer进行训练
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./tmp_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
    save_strategy="steps",
    save_steps=20,
)
trainer = SFTTrainer(
    model=original_model,
    tokenizer=tokenizer,
    train_dataset=combined_dataset,
    args=training_args,
    dataset_text_field="text",
)


#显示当前内存统计
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#开始训练模型
# 要恢复训练，可设置 resume_from_checkpoint = True
trainer_stats = trainer.train()

#保存LoRA适配器（不包含完整模型，体积小）
# 保存微调后的模型
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
logger.info("微调模型已保存")

# 首先确保模型处于评估模式
# 加载微调后的模型和分词器
fine_tuned_model = AutoModelForCausalLM.from_pretrained(
    "./fine_tuned_model",
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# 准备测试数据
test_conversations = fine_tuned_tokenizer.apply_chat_template(
    mapped_dataset["test"]["conversations"],
    tokenize=False,
)

# 创建测试数据集
test_data = pd.Series(test_conversations)
test_dataset = Dataset.from_pandas(pd.DataFrame({"text": test_data}))[0:100]  # 取前100条测试，避免内存不足

# 评估函数
def evaluate_model(test_dataset, model, tokenizer):
    correct = 0
    total = len(test_dataset)
    
    for example in test_dataset:
        # 获取问题和真实答案
        conversation = example["text"]
        # 通常最后一个用户问题是我们要回答的
        # 这里假设对话格式是固定的：[user, assistant]交替
        parts = conversation.split(tokenizer.eos_token)
        last_user_query = parts[-2] if len(parts) > 1 else parts[0]
        
        # 生成模型回答
        inputs = tokenizer(last_user_query, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码模型输出
        model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 获取真实答案 (从对话中提取)
        true_answer = parts[-1].split("assistant")[-1].strip() if len(parts) > 1 else ""
        
        # 简单匹配判断是否正确 (实际应用中可能需要更复杂的评估逻辑)
        if true_answer and true_answer in model_output:
            correct += 1
    
    accuracy = correct / total
    return accuracy

# 运行评估
accuracy = evaluate_model(test_dataset, original_model, tokenizer)
print(f"测试集准确率: {accuracy:.2%}")

# 更详细的评估示例（打印部分样本）
def detailed_evaluation(test_dataset, model, tokenizer, num_samples=5):
    for i, example in enumerate(test_dataset[:num_samples]):
        conversation = example["text"]
        parts = conversation.split(tokenizer.eos_token)
        question = parts[-2] if len(parts) > 1 else parts[0]
        true_answer = parts[-1].split("assistant")[-1].strip() if len(parts) > 1 else ""
        
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n示例 {i+1}:")
        print(f"问题: {question}")
        print(f"真实答案: {true_answer}")
        print(f"模型回答: {model_answer}")
        print("="*50)

# 运行详细评估
detailed_evaluation(test_dataset, original_model, tokenizer)