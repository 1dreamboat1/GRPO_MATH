

# from datasets import load_dataset

# ds = load_dataset("./gsm8k", "main")

# print(ds)


# from datasets import load_dataset
# import os
# # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# # # 加载一个公开数据集，例如 "imdb"
# dataset = load_dataset("imdb")
# print(dataset)

# from datasets import load_dataset

# ds = load_dataset("./gsm8k", "main")  # "main" 是默认的配置名称 也可以改为"socratic"

# print(ds)
"""
DatasetDict({
    train: Dataset({
        features: ['question', 'answer'],
        num_rows: 7473
    })
    test: Dataset({
        features: ['question', 'answer'],
        num_rows: 1319
    })
})
"""
# print(ds["train"][0])
"""
{'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 
'answer': 'How many clips did Natalia sell in May? ** Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nHow many clips did Natalia sell altogether in April and May? ** Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72'}


{'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 
'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72'}
"""

# print(ds["train"].features)
"""
{'question': Value(dtype='string', id=None), 
'answer': Value(dtype='string', id=None)}
"""
# print(ds['train'][0]['answer'])


# import os
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:8899"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8899"


# import requests

# try:
#     response = requests.get("https://huggingface.co", timeout=10)
#     print("连接成功")
# except Exception as e:
#     print(f"连接失败: {e}")


# Use a pipeline as a high-level helper
# from transformers import pipeline

# model_path = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
# pipe = pipeline("text-generation", model=model_path ,trust_remote_code=True)
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# response = pipe(messages,max_new_tokens=512)
# print(response)



# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2-0.5B-Instruct",
#     trust_remote_code=True  # 关键参数
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     "Qwen/Qwen2-0.5B-Instruct",
#     trust_remote_code=True
# )




# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_path = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"  # 替换为你实际的模型路径

# # 加载 tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# # 加载模型
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# print("Tokenizer and model loaded successfully.")


# zero_shot_results= {
#     "accuracy": 0.85,
#     "avg_partial_score": 0.75
# }
# few_shot_3_results = {
#     "accuracy": 0.90,
#     "avg_partial_score": 0.80
# }
# few_shot_5_results = {
#     "accuracy": 0.95,
#     "avg_partial_score": 0.85
# }
# # 汇总结果
# print(f"\n{'='*60}")
# print("汇总结果对比")
# print(f"{'='*60}")
# print(f"{'测试类型':<21} {'准确率':<16} {'部分得分':<10}")
# print(f"{'-'*60}")
# print(f"{'Zero-Shot':<25} {zero_shot_results['accuracy']:<20.4f} {zero_shot_results['avg_partial_score']:<20.4f}")
# print(f"{'3-Shot':<25} {few_shot_3_results['accuracy']:<20.4f} {few_shot_3_results['avg_partial_score']:<20.4f}")
# print(f"{'5-Shot':<25} {few_shot_5_results['accuracy']:<20.4f} {few_shot_5_results['avg_partial_score']:<20.4f}")
# def hello():print("Hello, World!")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_name = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 关键优化：启用所有加速选项
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    # attn_implementation="flash_attention_2",  # 必须安装flash-attn
    low_cpu_mem_usage=True  # 减少CPU内存拷贝
).eval()  # 禁用dropout等训练模式

start = time.time()
# 将tokenization也移到GPU（需支持GPU的tokenizer）
input_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 预热GPU（避免首次运行延迟）
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=1)

# 正式测速
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    # do_sample=True,
    do_sample=False,
    use_cache=True,  # 启用KV缓存
    pad_token_id=tokenizer.eos_token_id  # 避免padding计算
)
print(f"Time: {(time.time() - start):.3f}s")