import random
from typing import List, Dict, Optional
from datasets import load_dataset


class GSM8KDataset:
    """GSM8K数据集类"""
    
    def __init__(self, dataset_path: str = "/root/autodl-tmp/GRPO_MATH/gsm8k", type_name: str = "main"):
        """
        初始化GSM8K数据集
        
        Args:
            dataset_path: 数据集路径
            type_name: 数据集类型名称，可选 "main" 或 "socratic"
        """
        self.dataset_path = dataset_path
        self.type_name = type_name
        # self.dataset = None
        self.train_data = None
        self.test_data = None
        # self.load_data()  # 自动加载数据集
        
    def load_data(self):
        """加载数据集"""
        print(f"正在加载GSM8K数据集 (类型: {self.type_name})...")# 假设你放在 ./gsm8k/train.parquet 和 ./gsm8k/test.parquet
        main_train_path = f"{self.dataset_path}/main/train-00000-of-00001.parquet"
        main_test_path = f"{self.dataset_path}/main/test-00000-of-00001.parquet"
        socratic_train_path = f"{self.dataset_path}/socratic/train-00000-of-00001.parquet"
        socratic_test_path = f"{self.dataset_path}/socratic/test-00000-of-00001.parquet"
        self.dataset = load_dataset("parquet", data_files={
            "train": main_train_path,
            "test": main_test_path
        })
        # self.dataset = load_dataset(self.dataset_path, self.type_name)
        
        self.train_data = list(self.dataset['train'])
        self.test_data = list(self.dataset['test'])
        print(f"训练集大小: {len(self.train_data)}")
        print(f"测试集大小: {len(self.test_data)}")
        
    def get_train_samples(self, n_samples: Optional[int] = None) -> List[Dict]:
        """获取训练样本"""
        if self.train_data is None:
            self.load_data()
        
        if n_samples is None:
            return self.train_data
        return random.sample(self.train_data, min(n_samples, len(self.train_data)))
    
    def get_test_samples(self, n_samples: Optional[int] = None) -> List[Dict]:
        """获取测试样本"""
        if self.test_data is None:
            self.load_data()
            
        if n_samples is None:
            return self.test_data
        # return random.sample(self.test_data, min(n_samples, len(self.test_data)))
        return self.test_data[:min(n_samples, len(self.test_data))]
    
    def get_few_shot_examples(self, n_shots: int) -> List[Dict]:
        """获取few-shot示例"""
        return self.get_train_samples(n_shots)
