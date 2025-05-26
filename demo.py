import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_dataset
import random
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GSM8KDataset(Dataset):
    """GSM8K数据集处理类
    
    这个类负责处理GSM8K数学问题数据集，将原始的问题答案对
    转换为适合强化学习训练的格式，特别保留推理链信息
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_and_process_data(data_path)
    
    def _load_and_process_data(self, data_path: str) -> List[Dict]:
        """加载并预处理GSM8K数据
        
        将原始数据转换为包含问题、推理步骤和答案的结构化格式
        """
        try:
            # 加载GSM8K数据集
            dataset = load_dataset("gsm8k", "main")
            train_data = dataset['train']
            
            processed_data = []
            for item in train_data:
                question = item['question']
                answer = item['answer']
                
                # 提取最终数字答案
                final_answer = self._extract_final_answer(answer)
                
                # 构造CoT格式的完整响应
                full_response = f"让我逐步解决这个问题：\n\n{answer}"
                
                processed_data.append({
                    'question': question,
                    'answer': answer,
                    'final_answer': final_answer,
                    'full_response': full_response
                })
            
            logger.info(f"成功加载 {len(processed_data)} 条训练数据")
            return processed_data
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            # 如果无法加载真实数据，创建示例数据
            return self._create_sample_data()
    
    def _extract_final_answer(self, answer_text: str) -> str:
        """从答案文本中提取最终的数字答案"""
        # 查找形如 "#### 数字" 的模式
        pattern = r'####\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, answer_text)
        if match:
            return match.group(1)
        
        # 如果没找到，尝试提取最后出现的数字
        numbers = re.findall(r'\d+(?:\.\d+)?', answer_text)
        return numbers[-1] if numbers else "0"
    
    def _create_sample_data(self) -> List[Dict]:
        """创建示例数据用于演示"""
        sample_data = [
            {
                'question': "Janet has 3 apples and buys 2 more. How many apples does she have now?",
                'answer': "Janet starts with 3 apples.\nShe buys 2 more apples.\n3 + 2 = 5\n#### 5",
                'final_answer': "5",
                'full_response': "让我逐步解决这个问题：\n\nJanet starts with 3 apples.\nShe buys 2 more apples.\n3 + 2 = 5\n#### 5"
            },
            {
                'question': "A school has 4 classes with 25 students each. How many students are there in total?",
                'answer': "Each class has 25 students.\nThere are 4 classes.\n25 × 4 = 100\n#### 100",
                'final_answer': "100",
                'full_response': "让我逐步解决这个问题：\n\nEach class has 25 students.\nThere are 4 classes.\n25 × 4 = 100\n#### 100"
            }
        ]
        
        # 复制数据以创建更大的数据集
        extended_data = []
        for _ in range(50):  # 创建100条数据用于演示
            extended_data.extend(sample_data)
        
        logger.info(f"创建了 {len(extended_data)} 条示例数据")
        return extended_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构造输入文本
        prompt = f"问题: {item['question']}\n答案: "
        full_text = prompt + item['full_response']
        
        # 编码文本
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'question': item['question'],
            'expected_answer': item['final_answer'],
            'full_response': item['full_response']
        }

class RewardFunction:
    """奖励函数类
    
    设计多维度的奖励函数来评估模型生成的答案质量
    包括正确性奖励、格式奖励、推理质量奖励等
    """
    
    def __init__(self):
        self.weights = {
            'correctness': 1.0,    # 答案正确性权重
            'format': 0.3,         # 格式规范性权重
            'reasoning': 0.5       # 推理质量权重
        }
    
    def calculate_reward(self, question: str, generated_answer: str, expected_answer: str) -> Dict[str, float]:
        """计算综合奖励分数"""
        
        # 正确性奖励
        correctness_reward = self._calculate_correctness_reward(generated_answer, expected_answer)
        
        # 格式奖励
        format_reward = self._calculate_format_reward(generated_answer)
        
        # 推理质量奖励
        reasoning_reward = self._calculate_reasoning_reward(generated_answer)
        
        # 计算总奖励
        total_reward = (
            self.weights['correctness'] * correctness_reward +
            self.weights['format'] * format_reward +
            self.weights['reasoning'] * reasoning_reward
        )
        
        return {
            'total_reward': total_reward,
            'correctness': correctness_reward,
            'format': format_reward,
            'reasoning': reasoning_reward
        }
    
    def _calculate_correctness_reward(self, generated: str, expected: str) -> float:
        """计算答案正确性奖励"""
        # 提取生成答案中的数字
        generated_numbers = re.findall(r'\d+(?:\.\d+)?', generated)
        expected_num = float(expected) if expected.replace('.', '').isdigit() else 0
        
        if not generated_numbers:
            return 0.0
        
        # 检查最后一个数字是否与期望答案匹配
        try:
            last_number = float(generated_numbers[-1])
            if abs(last_number - expected_num) < 1e-6:
                return 1.0
            else:
                # 给予部分奖励，基于数值的接近程度
                diff = abs(last_number - expected_num)
                max_diff = max(expected_num, 1.0)  # 避免除零
                return max(0.0, 1.0 - diff / max_diff)
        except ValueError:
            return 0.0
    
    def _calculate_format_reward(self, generated: str) -> float:
        """计算格式规范性奖励"""
        reward = 0.0
        
        # 检查是否包含计算步骤
        if '=' in generated:
            reward += 0.3
        
        # 检查是否有逐步推理
        if any(keyword in generated.lower() for keyword in ['首先', '然后', '接下来', 'first', 'then', 'next']):
            reward += 0.3
        
        # 检查是否有最终答案标识
        if '####' in generated or '答案' in generated or 'answer' in generated.lower():
            reward += 0.4
        
        return min(reward, 1.0)
    
    def _calculate_reasoning_reward(self, generated: str) -> float:
        """计算推理质量奖励"""
        reward = 0.0
        
        # 检查推理步骤的数量
        steps = generated.count('\n')
        if steps >= 2:
            reward += 0.4
        
        # 检查是否包含数学运算
        math_operations = ['+', '-', '×', '*', '÷', '/']
        if any(op in generated for op in math_operations):
            reward += 0.3
        
        # 检查逻辑连贯性（简化版本）
        if len(generated.split()) >= 10:  # 至少有一定长度的解释
            reward += 0.3
        
        return min(reward, 1.0)

class GRPOTrainer:
    """GRPO训练器类
    
    实现Group Relative Policy Optimization算法的核心逻辑
    通过比较不同策略生成的样本来优化模型
    """
    
    def __init__(self, model, tokenizer, reward_function, lr=1e-5, beta=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.beta = beta  # KL散度正则化系数
        
        # 训练状态追踪
        self.training_stats = {
            'rewards': [],
            'kl_divergences': [],
            'policy_losses': [],
            'total_losses': []
        }
    
    def train_step(self, batch_questions: List[str], batch_expected: List[str]) -> Dict[str, float]:
        """执行一步GRPO训练"""
        
        # 生成多个候选答案
        candidate_responses = self._generate_candidates(batch_questions, num_candidates=4)
        
        # 计算奖励
        rewards = self._calculate_batch_rewards(batch_questions, candidate_responses, batch_expected)
        
        # 计算GRPO损失
        loss_info = self._compute_grpo_loss(batch_questions, candidate_responses, rewards)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss_info['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 记录统计信息
        self._update_stats(loss_info, rewards)
        
        return {
            'total_loss': loss_info['total_loss'].item(),
            'policy_loss': loss_info['policy_loss'].item(),
            'avg_reward': np.mean([r['total_reward'] for batch_r in rewards for r in batch_r])
        }
    
    def _generate_candidates(self, questions: List[str], num_candidates: int = 4) -> List[List[str]]:
        """为每个问题生成多个候选答案"""
        all_candidates = []
        
        for question in questions:
            candidates = []
            prompt = f"问题: {question}\n答案: "
            
            for _ in range(num_candidates):
                inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码生成的文本
                generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                candidates.append(generated.strip())
            
            all_candidates.append(candidates)
        
        return all_candidates
    
    def _calculate_batch_rewards(self, questions: List[str], candidates: List[List[str]], expected: List[str]) -> List[List[Dict]]:
        """计算批次中所有候选答案的奖励"""
        batch_rewards = []
        
        for q, cands, exp in zip(questions, candidates, expected):
            candidate_rewards = []
            for candidate in cands:
                reward_info = self.reward_function.calculate_reward(q, candidate, exp)
                candidate_rewards.append(reward_info)
            batch_rewards.append(candidate_rewards)
        
        return batch_rewards
    
    def _compute_grpo_loss(self, questions: List[str], candidates: List[List[str]], rewards: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """计算GRPO损失函数"""
        total_loss = 0.0
        policy_loss = 0.0
        kl_loss = 0.0
        
        for q, cands, cand_rewards in zip(questions, candidates, rewards):
            # 计算每个候选答案的对数概率
            log_probs = self._compute_log_probabilities(q, cands)
            
            # 提取奖励值
            reward_values = torch.tensor([r['total_reward'] for r in cand_rewards], dtype=torch.float32)
            
            # GRPO的核心：使用相对奖励进行策略优化
            # 计算相对优势
            mean_reward = reward_values.mean()
            advantages = reward_values - mean_reward
            
            # 策略损失：鼓励高奖励的动作，惩罚低奖励的动作
            policy_loss_batch = -(log_probs * advantages).mean()
            
            policy_loss += policy_loss_batch
        
        total_loss = policy_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'kl_loss': kl_loss
        }
    
    def _compute_log_probabilities(self, question: str, candidates: List[str]) -> torch.Tensor:
        """计算候选答案的对数概率"""
        log_probs = []
        
        for candidate in candidates:
            prompt = f"问题: {question}\n答案: "
            full_text = prompt + candidate
            
            inputs = self.tokenizer(full_text, return_tensors='pt', padding=True)
            
            with torch.no_grad():
                outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                logits = outputs.logits
                
                # 计算候选答案部分的对数概率
                prompt_length = len(self.tokenizer(prompt)['input_ids'])
                answer_logits = logits[0, prompt_length-1:-1, :]
                answer_tokens = inputs['input_ids'][0, prompt_length:]
                
                log_prob = torch.nn.functional.log_softmax(answer_logits, dim=-1)
                token_log_probs = log_prob.gather(1, answer_tokens.unsqueeze(1)).squeeze()
                total_log_prob = token_log_probs.sum()
                
                log_probs.append(total_log_prob)
        
        return torch.stack(log_probs)
    
    def _update_stats(self, loss_info: Dict[str, torch.Tensor], rewards: List[List[Dict]]):
        """更新训练统计信息"""
        avg_reward = np.mean([r['total_reward'] for batch_r in rewards for r in batch_r])
        
        self.training_stats['rewards'].append(avg_reward)
        self.training_stats['total_losses'].append(loss_info['total_loss'].item())
        self.training_stats['policy_losses'].append(loss_info['policy_loss'].item())

class ModelEvaluator:
    """模型评估器类
    
    负责评估不同阶段模型的性能，包括准确率计算和结果分析
    """
    
    def __init__(self, tokenizer, reward_function):
        self.tokenizer = tokenizer
        self.reward_function = reward_function
    
    def evaluate_model(self, model, test_dataset: GSM8KDataset, num_samples: int = 50) -> Dict[str, float]:
        """评估模型在测试集上的性能"""
        model.eval()
        
        correct_predictions = 0
        total_predictions = 0
        total_reward = 0.0
        detailed_results = []
        
        # 随机选择测试样本
        test_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
        
        for idx in tqdm(test_indices, desc="评估模型性能"):
            sample = test_dataset[idx]
            question = sample['question']
            expected_answer = sample['expected_answer']
            
            # 生成模型答案
            generated_answer = self._generate_answer(model, question)
            
            # 计算奖励和准确性
            reward_info = self.reward_function.calculate_reward(question, generated_answer, expected_answer)
            is_correct = reward_info['correctness'] > 0.8  # 准确性阈值
            
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            total_reward += reward_info['total_reward']
            
            detailed_results.append({
                'question': question,
                'expected': expected_answer,
                'generated': generated_answer,
                'reward': reward_info,
                'correct': is_correct
            })
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_reward = total_reward / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'avg_reward': avg_reward,
            'total_samples': total_predictions,
            'detailed_results': detailed_results
        }
    
    def _generate_answer(self, model, question: str) -> str:
        """使用模型生成问题的答案"""
        prompt = f"问题: {question}\n答案: "
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated.strip()

def plot_training_results(training_stats: Dict, evaluation_results: Dict):
    """可视化训练结果和性能对比"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励变化图
    axes[0, 0].plot(training_stats['rewards'])
    axes[0, 0].set_title('平均奖励随训练步数变化')
    axes[0, 0].set_xlabel('训练步数')
    axes[0, 0].set_ylabel('平均奖励')
    axes[0, 0].grid(True)
    
    # 损失变化图
    axes[0, 1].plot(training_stats['total_losses'], label='总损失')
    axes[0, 1].plot(training_stats['policy_losses'], label='策略损失')
    axes[0, 1].set_title('损失函数变化')
    axes[0, 1].set_xlabel('训练步数')
    axes[0, 1].set_ylabel('损失值')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 模型性能对比
    models = list(evaluation_results.keys())
    accuracies = [evaluation_results[model]['accuracy'] for model in models]
    
    axes[1, 0].bar(models, accuracies)
    axes[1, 0].set_title('不同模型准确率对比')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 奖励分布图
    rewards = [evaluation_results[model]['avg_reward'] for model in models]
    axes[1, 1].bar(models, rewards)
    axes[1, 1].set_title('不同模型平均奖励对比')
    axes[1, 1].set_ylabel('平均奖励')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数：执行完整的GRPO强化学习微调流程"""
    
    print("开始GRPO强化学习微调实验...")
    
    # Step 1: 数据准备
    print("\nStep 1: 数据准备")
    
    # 初始化tokenizer (使用较小的模型进行演示)
    model_name = "microsoft/DialoGPT-small"  # 使用更容易获得的模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        print("无法加载预训练模型，使用模拟tokenizer")
        # 创建简单的模拟tokenizer用于演示
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
                self.eos_token_id = 0
            
            def __call__(self, text, **kwargs):
                # 简单的模拟编码
                tokens = text.split()
                input_ids = list(range(len(tokens)))
                attention_mask = [1] * len(tokens)
                
                if kwargs.get('padding') == 'max_length':
                    max_len = kwargs.get('max_length', 512)
                    while len(input_ids) < max_len:
                        input_ids.append(0)
                        attention_mask.append(0)
                
                result = {
                    'input_ids': torch.tensor([input_ids]),
                    'attention_mask': torch.tensor([attention_mask])
                }
                
                if kwargs.get('return_tensors') == 'pt':
                    return result
                return {'input_ids': input_ids, 'attention_mask': attention_mask}
            
            def decode(self, tokens, **kwargs):
                return f"生成的答案 {len(tokens)} tokens"
        
        tokenizer = MockTokenizer()
    
    # 加载数据集
    dataset = GSM8KDataset("", tokenizer)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    print(f"训练集大小: {len(train_indices)}")
    print(f"验证集大小: {len(val_indices)}")
    
    # Step 2: 模型训练
    print("\nStep 2: 模型初始化和训练")
    
    # 创建模拟模型用于演示
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 1000)  # 简单的线性层
            
        def forward(self, input_ids, attention_mask=None):
            # 模拟输出
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, 1000)
            return type('obj', (object,), {'logits': logits})
        
        def generate(self, input_ids, **kwargs):
            # 模拟生成
            batch_size, seq_len = input_ids.shape
            new_tokens = torch.randint(0, 1000, (batch_size, kwargs.get('max_new_tokens', 50)))
            return torch.cat([input_ids, new_tokens], dim=1)
        
        def parameters(self):
            return self.linear.parameters()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except:
        print("使用模拟模型进行演示")
        model = MockModel()
    
    # 初始化奖励函数和评估器
    reward_function = RewardFunction()
    evaluator = ModelEvaluator(tokenizer, reward_function)
    
    # Step 3: 基线评估
    print("\nStep 3: 基线模型评估")
    baseline_results = evaluator.evaluate_model(model, dataset, num_samples=10)
    print(f"基线模型准确率: {baseline_results['accuracy']:.3f}")
    print(f"基线模型平均奖励: {baseline_results['avg_reward']:.3f}")
    
    # Step 4: GRPO训练
    print("\nStep 4: GRPO强化学习训练")
    grpo_trainer = GRPOTrainer(model, tokenizer, reward_function)
    
    # 模拟训练过程
    num_epochs = 5
    batch_size = 2
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 随机选择训练样本
        epoch_indices = random.sample(train_indices, min(10, len(train_indices)))
        
        for i in range(0, len(epoch_indices), batch_size):
            batch_indices = epoch_indices[i:i+batch_size]
            batch_questions = [dataset[idx]['question'] for idx in batch_indices]
            batch_expected = [dataset[idx]['expected_answer'] for idx in batch_indices]
            
            # 执行训练步骤
            step_results = grpo_trainer.train_step(batch_questions, batch_expected)
            
            if i % (batch_size * 2) == 0:  # 每隔几个batch打印一次
                print(f"  Batch {i//batch_size + 1}: Loss={step_results['total_loss']:.4f}, Reward={step_results['avg_reward']:.3f}")
    
    # Step 5: 训练后评估
    print("\nStep 5: 训练后模型评估")
    final_results = evaluator.evaluate_model(model, dataset, num_samples=10)
    print(f"GRPO训练后准确率: {final_results['accuracy']:.3f}")
    print(f"GRPO训练后平均奖励: {final_results['avg_reward']:.3f}")
    
    # Step 6: 结果对比和可视化
    # print("\nStep