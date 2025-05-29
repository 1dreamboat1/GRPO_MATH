import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm import tqdm
import re

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path: str = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
dataset_path: str = "/root/autodl-tmp/GRPO_MATH/gsm8k"
# 加载 Qwen2-0.5B-Instruct 模型（4-bit 量化）
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True
# )

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # quantization_config=quant_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载 GSM8K 测试集
main_test_path = f"{dataset_path}/main/test-00000-of-00001.parquet"
dataset = load_dataset("parquet", data_files={
            "test": main_test_path
        })
dataset = list(dataset['test'])[:20]
# dataset = load_dataset("gsm8k", "main")["test"]

# 定义 CoT 提示模板
prompt_template = """Solve the following math problem step-by-step. Provide the final answer as a number.

Problem: {question}

Step-by-step reasoning:
"""

def extract_final_answer(response):
    """从模型输出中提取最终数字答案"""
    # 查找最后出现的数字（包括小数和负数）
    numbers = re.findall(r'-?\d+\.?\d*', response)
    return float(numbers[-1]) if numbers else None

# 用于存储结果
results = []
correct_count = 0
total_questions = len(dataset)

# 遍历测试集
for item in tqdm(dataset, desc="Processing GSM8K questions"):
    question = item["question"]
    correct_answer = float(item["answer"].split("#### ")[1])
    
    # 构造提示
    prompt = prompt_template.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 生成回答
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.1,
        top_p=0.1,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取最终答案
    predicted_answer = extract_final_answer(response)
    
    # 判断是否正确
    is_correct = predicted_answer is not None and abs(predicted_answer - correct_answer) < 1e-5
    
    # 记录结果
    result = {
        "question": question,
        "model_answer": response,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct
    }
    results.append(result)
    
    if is_correct:
        correct_count += 1

# 计算准确率
accuracy = (correct_count / total_questions) * 100

# 保存结果到文件
with open("gsm8k_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "results": results,
        "accuracy": accuracy,
        "total_questions": total_questions,
        "correct_count": correct_count
    }, f, ensure_ascii=False, indent=4)

# 打印准确率
print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_questions})")
print("Results saved to gsm8k_results.json")