import os
import re
import json
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from datasets import load_dataset, Dataset
from trl import PPOConfig, PPOTrainer, set_seed as trl_set_seed

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RLTrainingConfig:
    """强化学习训练配置类"""
    # 路径配置
    base_model_path: str = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
    lora_model_path: str = "./lora_finetuned_qwen"
    dataset_path: str = "/root/autodl-tmp/GRPO_MATH/gsm8k"
    output_dir: str = "./rl_finetuned_qwen"
    
    # 数据参数
    max_length: int = 512
    train_val_split_ratio: float = 0.9
    
    # PPO训练参数
    batch_size: int = 2
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # PPO特定参数
    ppo_epochs: int = 4       # PPO更新次数
    init_kl_coef: float = 0.2 # KL散度初始系数
    target_kl: float = 6.0    # 目标KL散度
    gamma: float = 0.99       # 折扣因子
    lam: float = 0.95         # GAE-Lambda
    cliprange: float = 0.2    # PPO裁剪参数
    cliprange_value: float = 0.2  # 价值函数裁剪参数
    vf_coef: float = 0.1      # 价值损失系数
    
    # 奖励函数参数
    correctness_reward: float = 10.0  # 答案正确的奖励
    format_reward: float = 2.0  # 格式正确的奖励
    step_reward: float = 1.0  # 每个推理步骤的奖励
    
    # LoRA参数
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])
    
    # 其他配置
    use_bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    seed: int = 42
    
    def __post_init__(self):
        """后处理配置"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

class GSM8KDataHandler:
    """GSM8K数据集处理器"""
    
    def __init__(self, config: RLTrainingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.train_data = None
        self.test_data = None
    
    def extract_answer_from_text(self, answer_text: str) -> Optional[float]:
        """从文本中提取最终答案"""
        # 寻找 #### 后面的数字
        pattern = r'####\s*(-?\d+(?:\.\d+)?)'
        match = re.search(pattern, answer_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # 如果没找到####，寻找最后一个数字
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def has_correct_format(self, text: str) -> bool:
        """检查文本是否包含正确的格式"""
        # 检查是否包含 #### 标记
        has_marker = "####" in text
        
        # 检查是否包含多个推理步骤（至少有3行文本）
        has_steps = len(text.split('\n')) >= 3
        
        return has_marker and has_steps
    
    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """加载GSM8K数据集"""
        logger.info("开始加载GSM8K数据集...")
        
        try:
            # 加载Parquet格式
            train_parquet_path = os.path.join(self.config.dataset_path, "main/train-00000-of-00001.parquet")
            test_parquet_path = os.path.join(self.config.dataset_path, "main/test-00000-of-00001.parquet")
            
            if os.path.exists(train_parquet_path):
                logger.info(f"从Parquet文件加载数据集")
                dataset = load_dataset("parquet", data_files={
                    "train": train_parquet_path,
                    "test": test_parquet_path
                })
                self.train_data = list(dataset['train'])
                self.test_data = list(dataset['test'])
            
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
        
        logger.info(f"训练集大小: {len(self.train_data)}")
        logger.info(f"测试集大小: {len(self.test_data)}")
        
        # 为GRPO格式化数据
        processed_train_data = self._preprocess_data_for_grpo(self.train_data)
        
        # 创建Dataset对象并划分
        full_dataset = Dataset.from_list(processed_train_data)
        dataset_split = full_dataset.train_test_split(
            test_size=1 - self.config.train_val_split_ratio, 
            seed=self.config.seed
        )
        
        train_dataset = dataset_split['train']
        val_dataset = dataset_split['test']
        
        logger.info(f"划分后训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def _preprocess_data_for_grpo(self, raw_data: List[Dict]) -> List[Dict]:
        """为GRPO预处理数据"""
        processed_data = []
        
        for sample in tqdm(raw_data, desc="为GRPO预处理数据"):
            question = sample['question']
            answer = sample['answer']
            answer_label = self.extract_answer_from_text(answer)
            
            # 格式化为GRPO训练所需的格式
            processed_sample = {
                "question": question,
                "answer": answer,
                "answer_label": answer_label,
                "prompt": f"Please solve this math problem step by step and provide your final answer after the \"####\" marker.\n{question}",
            }
            
            # 跳过无法提取答案的样本
            if answer_label is None:
                continue
                
            processed_data.append(processed_sample)
        
        logger.info(f"GRPO数据预处理完成，样本数量: {len(processed_data)}")
        
        # 展示几个样本
        for i in range(min(3, len(processed_data))):
            logger.info(f"\n=== 样本 {i+1} ===")
            logger.info(f"问题: {processed_data[i]['question']}")
            logger.info(f"答案: {processed_data[i]['answer']}")
            logger.info(f"答案标签: {processed_data[i]['answer_label']}")
            logger.info(f"提示: {processed_data[i]['prompt']}")
            
        return processed_data
    
    def get_test_samples(self, n_samples: Optional[int] = None) -> List[Dict]:
        """获取测试样本"""
        if self.test_data is None:
            raise ValueError("测试数据未加载，请先调用load_dataset()")
            
        if n_samples is None:
            return self.test_data
        return self.test_data[:min(n_samples, len(self.test_data))]

class RewardCalculator:
    """奖励计算器，用于计算模型生成的回答的奖励"""
    
    def __init__(self, config: RLTrainingConfig, data_handler: GSM8KDataHandler):
        self.config = config
        self.data_handler = data_handler
    
    def compute_rewards(self, questions: List[str], 
                        generated_texts: List[str], 
                        ground_truth_answers: List[float]) -> List[float]:
        """计算一批回答的奖励"""
        rewards = []
        
        for question, text, gt_answer in zip(questions, generated_texts, ground_truth_answers):
            # 1. 计算正确性奖励
            predicted_answer = self.data_handler.extract_answer_from_text(text)
            correctness_reward = 0.0
            
            if predicted_answer is not None and gt_answer is not None:
                # 允许小的数值误差
                if abs(predicted_answer - gt_answer) < 1e-6:
                    correctness_reward = self.config.correctness_reward
                else:
                    # 给予部分奖励，基于相对误差
                    relative_error = abs(predicted_answer - gt_answer) / max(1e-6, abs(gt_answer))
                    if relative_error < 0.1:  # 误差小于10%
                        correctness_reward = self.config.correctness_reward * (1 - relative_error)
            
            # 2. 计算格式奖励
            format_reward = 0.0
            if self.data_handler.has_correct_format(text):
                format_reward = self.config.format_reward
            
            # 3. 计算步骤奖励
            step_reward = 0.0
            lines = [line for line in text.split('\n') if line.strip()]
            # 每个非空行视为一个步骤，但限制奖励上限
            step_count = min(len(lines) - 1, 5)  # 减去问题行，最多奖励5个步骤
            if step_count > 0:
                step_reward = self.config.step_reward * step_count
            
            # 计算总奖励
            total_reward = correctness_reward + format_reward + step_reward
            rewards.append(total_reward)
            
        return rewards

class RLModelTrainer:
    """强化学习模型训练器"""
    
    def __init__(self, config: RLTrainingConfig):
        self.config = config
        self._setup_environment()
        self.tokenizer = None
        self.model = None
        self.ref_model = None  # 参考模型，用于KL散度计算
        self.data_handler = None
        self.reward_calculator = None
        
    def _setup_environment(self):
        """设置环境变量和随机种子"""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        set_seed(self.config.seed)
        trl_set_seed(self.config.seed)
        
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info(f"加载基础模型: {self.config.base_model_path}")
        logger.info(f"加载LoRA适配器: {self.config.lora_model_path}")
        
        # 首先尝试从LoRA适配器目录加载分词器
        try:
            logger.info(f"尝试从LoRA适配器目录加载分词器: {self.config.lora_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.lora_model_path,
                trust_remote_code=True, 
                padding_side='left'
            )
            logger.info("成功从LoRA适配器目录加载分词器")
        except Exception as e:
            logger.warning(f"从LoRA适配器目录加载分词器失败: {e}")
            logger.info(f"从基础模型加载分词器: {self.config.base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path, 
                trust_remote_code=True, 
                padding_side='left'
            )
    
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16
        )
        
        # 加载参考模型（用于KL散度计算）
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16
        )
        
        # 加载LoRA适配器
        try:
            peft_config = PeftConfig.from_pretrained(self.config.lora_model_path)
            self.model = PeftModel.from_pretrained(base_model, self.config.lora_model_path)
            logger.info(f"成功从 {self.config.lora_model_path} 加载LoRA适配器")
        except Exception as e:
            logger.warning(f"无法加载现有LoRA适配器: {e}，创建新的LoRA配置")
            # 如果无法加载，创建新的LoRA配置
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(base_model, lora_config)
        
        # 打印模型信息
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型总参数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """准备训练数据集"""
        self.data_handler = GSM8KDataHandler(self.config, self.tokenizer)
        train_dataset, val_dataset = self.data_handler.load_dataset()
        
        # 创建奖励计算器
        self.reward_calculator = RewardCalculator(self.config, self.data_handler)
        
        return train_dataset, val_dataset
    
    def _collator(self, data):
        """数据整理函数，用于批处理"""
        prompts = [d["prompt"] for d in data]
        questions = [d["question"] for d in data]
        answer_labels = [d["answer_label"] for d in data]
        
        return {
            "input_ids": prompts,
            "questions": questions,
            "answer_labels": answer_labels
        }
    
    def compute_reward(self, question_tensors, response_tensors, ground_truth_answers):
        """计算奖励"""
        batch_size = len(question_tensors)
        rewards = torch.zeros(batch_size, device=self.model.device)
        
        # 解码生成的回答
        questions = [self.tokenizer.decode(q, skip_special_tokens=True) for q in question_tensors]
        responses = [self.tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
        
        # 计算奖励
        reward_list = self.reward_calculator.compute_rewards(
            questions, 
            responses, 
            ground_truth_answers
        )
        
        # 转换为张量
        for i, reward in enumerate(reward_list):
            rewards[i] = reward
            
        return rewards
    
    def train(self):
        """执行强化学习训练"""
        logger.info("开始强化学习训练流程...")
        
        try:
            # 1. 加载模型和分词器
            self.load_model_and_tokenizer()
            
            # 2. 准备数据集
            train_dataset, val_dataset = self.prepare_datasets()
            
            # 3. 设置PPO训练参数
            ppo_config = PPOConfig(
                model_name=self.config.base_model_path,
                learning_rate=self.config.learning_rate,
                mini_batch_size=self.config.mini_batch_size,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                ppo_epochs=self.config.ppo_epochs,
                init_kl_coef=self.config.init_kl_coef,
                target_kl=self.config.target_kl,
                gamma=self.config.gamma,
                lam=self.config.lam,
                cliprange=self.config.cliprange,
                cliprange_value=self.config.cliprange_value,
                vf_coef=self.config.vf_coef,
                seed=self.config.seed,
            )
            
            # 4. 创建PPO训练器
            trainer = PPOTrainer(
                config=ppo_config,
                model=self.model,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
                dataset=train_dataset,
                data_collator=self._collator,
            )
            
            # 5. 清理显存
            torch.cuda.empty_cache()
            
            # 6. 开始训练
            logger.info("开始PPO训练...")
            
            # PPO训练循环
            for epoch in range(self.config.num_train_epochs):
                logger.info(f"开始epoch {epoch+1}/{self.config.num_train_epochs}")
                
                # 对每个batch进行训练
                for batch_idx, batch in enumerate(trainer.dataloader):
                    # 从batch获取数据
                    query_tensors = batch["input_ids"]
                    questions = batch["questions"]
                    answer_labels = batch["answer_labels"]
                    
                    # 使用模型生成回答
                    response_tensors = []
                    for query in query_tensors:
                        # 将query转换为tensor
                        query_tensor = torch.tensor(query if isinstance(query, list) else [query], device=trainer.accelerator.device)
                        # 生成回答
                        response = trainer.generate(
                            query_tensor, 
                            max_new_tokens=self.config.max_length//2,
                            do_sample=True,
                            temperature=0.7
                        )
                        response_tensors.append(response.squeeze())
                    
                    # 计算奖励
                    rewards = self.compute_reward(query_tensors, response_tensors, answer_labels)
                    
                    # 执行PPO更新
                    stats = trainer.step(query_tensors, response_tensors, rewards)
                    
                    # 记录训练信息
                    if batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: reward={stats['ppo/mean_rewards']:.4f}, kl={stats['ppo/mean_kl']:.4f}")
                
                # 每个epoch结束后保存模型
                output_dir = os.path.join(self.config.output_dir, f"epoch_{epoch+1}")
                os.makedirs(output_dir, exist_ok=True)
                trainer.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                logger.info(f"Epoch {epoch+1} 完成，模型已保存到 {output_dir}")
            
            # 7. 保存最终模型
            logger.info("保存最终模型...")
            trainer.model.save_pretrained(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            logger.info("强化学习训练完成!")
            
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def evaluate_model(self, sample_size: int = 100):
        """评估GRPO微调后的模型性能"""
        logger.info(f"开始评估GRPO微调后的模型性能 (样本数: {sample_size})...")
        
        # 确保数据处理器已初始化
        if self.data_handler is None:
            self.data_handler = GSM8KDataHandler(self.config, self.tokenizer)
            # 加载数据
            if self.data_handler.test_data is None:
                self.data_handler.load_dataset()
        
        # 获取测试样本
        test_samples = self.data_handler.get_test_samples(sample_size)
        logger.info(f"获取测试样本数量: {len(test_samples)}")
        
        # 准备评估数据
        prompts = []
        for sample in test_samples:
            prompt = f"Please solve this math problem step by step and provide your final answer after the \"####\" marker.\n{sample['question']}"
            prompts.append(prompt)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 处理结果
        results = []
        correct = 0
        format_correct = 0
        
        # 批量处理生成
        for i in tqdm(range(0, len(prompts), 8), desc="评估模型"):
            batch_prompts = prompts[i:i+8]
            batch_samples = test_samples[i:i+8]
            
            # 编码输入
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.max_length // 2  # 预留一半长度给生成
            ).to(self.model.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_length // 2,
                    temperature=0.1,
                    do_sample=False,  # 评估使用贪婪解码
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            # 解码回答
            for j, output in enumerate(outputs):
                sample = batch_samples[j]
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                
                # 移除提示部分
                original_prompt = batch_prompts[j]
                if response.startswith(original_prompt):
                    response = response[len(original_prompt):].strip()
                
                # 提取答案
                predicted_answer = self.data_handler.extract_answer_from_text(response)
                ground_truth_answer = self.data_handler.extract_answer_from_text(sample['answer'])
                
                # 检查格式是否正确
                has_correct_format = self.data_handler.has_correct_format(response)
                if has_correct_format:
                    format_correct += 1
                
                # 检查答案是否正确
                is_correct = (predicted_answer is not None and 
                             ground_truth_answer is not None and 
                             abs(predicted_answer - ground_truth_answer) < 1e-6)
                
                if is_correct:
                    correct += 1
                
                # 记录结果
                results.append({
                    'question': sample['question'],
                    'ground_truth': ground_truth_answer,
                    'predicted': predicted_answer,
                    'correct': is_correct,
                    'format_correct': has_correct_format,
                    'response': response
                })
                
                # 显示前几个样本的详细信息
                if i == 0 and j < 2:
                    logger.info(f"\n{'='*20} 测试样本 {j+1} {'='*20}")
                    logger.info(f"问题: {sample['question']}")
                    logger.info(f"模型回答:\n{response}")
                    logger.info(f"预测答案: {predicted_answer}")
                    logger.info(f"真实答案: {ground_truth_answer}")
                    logger.info(f"答案正确: {'✅' if is_correct else '❌'}")
                    logger.info(f"格式正确: {'✅' if has_correct_format else '❌'}")
            
            # 清理显存
            del inputs, outputs
            torch.cuda.empty_cache()
        
        # 计算评估指标
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        format_accuracy = format_correct / total if total > 0 else 0
        
        # 显示评估结果
        logger.info("\n" + "="*50)
        logger.info("GRPO微调后的模型评估结果")
        logger.info("="*50)
        logger.info(f"总样本数: {total}")
        logger.info(f"答案正确数: {correct}")
        logger.info(f"格式正确数: {format_correct}")
        logger.info(f"答案准确率: {accuracy:.4f} ({accuracy*100:.1f}%)")
        logger.info(f"格式准确率: {format_accuracy:.4f} ({format_accuracy*100:.1f}%)")
        
        # 保存评估结果
        eval_results = {
            "total_samples": total,
            "correct_predictions": correct,
            "format_correct": format_correct,
            "answer_accuracy": accuracy,
            "format_accuracy": format_accuracy,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "details": results
        }
        
        eval_path = os.path.join(self.config.output_dir, "grpo_evaluation_results.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"详细评估结果已保存到: {eval_path}")
        
        return eval_results

def save_config(config: RLTrainingConfig, filepath: str):
    """保存配置到文件"""
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def main():
    """主函数"""
    # 创建训练配置
    config = RLTrainingConfig(
        base_model_path="/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B",
        lora_model_path="./lora_finetuned_qwen",
        dataset_path="/root/autodl-tmp/GRPO_MATH/gsm8k",
        output_dir="./rl_finetuned_qwen",
        
        # PPO超参数
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-6,
        ppo_epochs=4,
        init_kl_coef=0.2,
        
        # 奖励函数权重
        correctness_reward=10.0,
        format_reward=2.0,
        step_reward=1.0
    )
    
    # 保存配置
    os.makedirs(config.output_dir, exist_ok=True)
    save_config(config, os.path.join(config.output_dir, "rl_config.json"))
    
    try:
        # 创建训练器
        trainer = RLModelTrainer(config)
        
        # 执行训练
        logger.info("="*50)
        logger.info("开始强化学习训练")
        logger.info("="*50)
        trainer.train()
        
        # 训练完成后评估模型
        logger.info("="*50)
        logger.info("强化学习训练完成，开始评估模型")
        logger.info("="*50)
        eval_results = trainer.evaluate_model(sample_size=100)
        
        # 显示最终结果摘要
        logger.info("="*50)
        logger.info("强化学习训练和评估完成！最终结果摘要：")
        logger.info("="*50)
        logger.info(f"✅ 模型训练完成，保存路径: {config.output_dir}")
        logger.info(f"📊 评估结果:")
        logger.info(f"   - 测试样本数: {eval_results['total_samples']}")
        logger.info(f"   - 答案正确数: {eval_results['correct_predictions']}")
        logger.info(f"   - 答案准确率: {eval_results['answer_accuracy']:.4f} ({eval_results['answer_accuracy']*100:.1f}%)")
        logger.info(f"   - 格式正确数: {eval_results['format_correct']}")
        logger.info(f"   - 格式准确率: {eval_results['format_accuracy']:.4f} ({eval_results['format_accuracy']*100:.1f}%)")
        logger.info(f"📄 详细评估报告保存至: {os.path.join(config.output_dir, 'rl_evaluation_results.json')}")
        
        logger.info("="*50)
        logger.info("所有任务完成！")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理CUDA缓存")

if __name__ == "__main__":
    main()
