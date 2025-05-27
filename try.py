

# from datasets import load_dataset

# ds = load_dataset("gsm8k", "main")

# print(ds)


# from datasets import load_dataset
# import os
# # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# # # 加载一个公开数据集，例如 "imdb"
# dataset = load_dataset("imdb")
# print(dataset)

# from datasets import load_dataset

# ds = load_dataset("./gsm8k", "main")  # "main" 是默认的配置名称 socratic

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


import requests

try:
    response = requests.get("https://huggingface.co", timeout=10)
    print("连接成功")
except Exception as e:
    print(f"连接失败: {e}")
