import json
import re
import random
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

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
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """加载数据集"""
        print(f"正在加载GSM8K数据集 (类型: {self.type_name})...")
        main_train_path = f"{self.dataset_path}/main/train-00000-of-00001.parquet"
        main_test_path = f"{self.dataset_path}/main/test-00000-of-00001.parquet"
        
        self.dataset = load_dataset("parquet", data_files={
            "train": main_train_path,
            "test": main_test_path
        })
        
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
        return random.sample(self.test_data, min(n_samples, len(self.test_data)))
    
    def get_few_shot_examples(self, n_shots: int) -> List[Dict]:
        """获取few-shot示例"""
        return self.get_train_samples(n_shots)

class QwenModel:
    """Qwen模型类 - 批量处理优化版本"""
    
    def __init__(self, model_path: str = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"):
        """
        初始化模型
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = None
        self._load_model()
        
    def _load_model(self):
        """加载模型和分词器"""
        print("正在加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True,padding_side='left')
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.device = next(self.model.parameters()).device
        print(f"模型已加载到设备: {self.device}")
        
    def generate_batch_responses(self, prompts: List[str], max_length: int = 512, 
                               temperature: float = 0.1, batch_size: int = 8) -> List[str]:
        """
        批量生成模型响应 - 核心优化点
        
        Args:
            prompts: 输入提示列表
            max_length: 最大生成长度
            temperature: 温度参数
            batch_size: 批处理大小
        """
        all_responses = []
        
        # 分批处理
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]
            
            # 批量编码
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512  # 限制输入长度防止显存不足
            ).to(self.device)
            
            with torch.no_grad():
                # 批量生成
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # 关键优化参数
                    use_cache=True,  # 使用缓存加速
                    num_beams=1,     # 使用贪婪解码加速
                )
            
            # 解码响应
            batch_responses = []
            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                # 移除输入提示部分
                original_prompt = batch_prompts[j]
                if response.startswith(original_prompt):
                    response = response[len(original_prompt):].strip()
                batch_responses.append(response)
            
            all_responses.extend(batch_responses)
            
            # 清理显存
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return all_responses
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.1) -> str:
        """单个响应生成（保持向后兼容）"""
        return self.generate_batch_responses([prompt], max_length, temperature, batch_size=1)[0]
    
    def create_zero_shot_prompt(self, question: str) -> str:
        """创建Zero-Shot提示"""
        return f"""Please solve this math problem step by step and provide your final answer after ####.

Question: {question}

Answer:"""
    
    def create_few_shot_prompt(self, question: str, examples: List[Dict]) -> str:
        """创建Few-Shot提示"""
        prompt = "Here are some examples of solving math problems:\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {example['question']}\n"
            prompt += f"Answer: {example['answer']}\n\n"
        
        prompt += f"Now solve this problem:\n"
        prompt += f"Question: {question}\n"
        prompt += f"Answer:"
        
        return prompt

class Tester:
    """测试类 - 批量处理优化版本"""
    
    def __init__(self, dataset: GSM8KDataset, model: QwenModel):
        """
        初始化测试器
        
        Args:
            dataset: 数据集对象
            model: 模型对象
        """
        self.dataset = dataset
        self.model = model
        
    def extract_answer(self, text: str) -> Optional[float]:
        """从文本中提取最终答案"""
        # 寻找 #### 后面的数字
        pattern = r'####\s*(-?\d+(?:\.\d+)?)'
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # 如果没找到####，寻找最后一个数字
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def extract_reasoning_steps(self, text: str) -> List[str]:
        """提取推理步骤"""
        steps = re.split(r'[.\n]', text)
        steps = [step.strip() for step in steps if len(step.strip()) > 10]
        return steps
    
    def calculate_accuracy(self, results: List[Dict]) -> float:
        """计算准确率"""
        if not results:
            return 0.0
        
        correct_count = sum(1 for result in results if result.get('correct', False))
        total_count = len(results)
        
        return correct_count / total_count
    
    def evaluate_partial_credit(self, predicted_steps: List[str], ground_truth: str) -> float:
        """评估部分得分"""
        if not predicted_steps:
            return 0.0
        
        gt_expressions = re.findall(r'<<([^>]+)>>', ground_truth)
        
        correct_steps = 0
        total_steps = len(predicted_steps)
        
        for step in predicted_steps:
            step_expressions = re.findall(r'(\d+(?:\.\d+)?)\s*[+\-*/]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', step)
            
            for expr in step_expressions:
                try:
                    a, op_match = float(expr[0]), re.search(r'[+\-*/]', step)
                    if op_match:
                        op = op_match.group()
                        b, result = float(expr[1]), float(expr[2])
                        
                        expected = None
                        if op == '+':
                            expected = a + b
                        elif op == '-':
                            expected = a - b
                        elif op == '*':
                            expected = a * b
                        elif op == '/':
                            expected = a / b if b != 0 else None
                        
                        if expected is not None and abs(expected - result) < 0.01:
                            correct_steps += 1
                            break
                except:
                    continue
            else:
                if any(keyword in step.lower() for keyword in ['because', 'so', 'therefore', 'thus', 'since']):
                    correct_steps += 0.5
        
        return correct_steps / total_steps if total_steps > 0 else 0.0
    
    def run_zero_shot_test(self, sample_size: int = 1319, batch_size: int = 16) -> Dict:
        """运行Zero-Shot测试 - 批量优化版本"""
        print(f"开始Zero-Shot测试 (批量大小: {batch_size})...")
        test_data = self.dataset.get_test_samples(sample_size)
        
        # 批量创建提示
        prompts = []
        for item in test_data:
            prompt = self.model.create_zero_shot_prompt(item['question'])
            prompts.append(prompt)
        
        # 批量生成响应
        print("正在批量生成响应...")
        responses = self.model.generate_batch_responses(prompts, batch_size=batch_size)
        
        # 处理结果
        results = []
        details = []
        
        print("正在处理结果...")
        for i, (item, response) in enumerate(tqdm(zip(test_data, responses), total=len(test_data))):
            question = item['question']
            ground_truth_answer = self.extract_answer(item['answer'])
            
            predicted_answer = self.extract_answer(response)
            predicted_steps = self.extract_reasoning_steps(response)
            partial_score = self.evaluate_partial_credit(predicted_steps, item['answer'])
            
            is_correct = (predicted_answer is not None and 
                         ground_truth_answer is not None and 
                         abs(predicted_answer - ground_truth_answer) < 0.01)
            
            result = {
                'correct': is_correct,
                'partial_score': partial_score
            }
            results.append(result)
            
            details.append({
                'question': question,
                'ground_truth': ground_truth_answer,
                'predicted': predicted_answer,
                'correct': is_correct,
                'partial_score': partial_score,
                'response': response
            })
        
        accuracy = self.calculate_accuracy(results)
        avg_partial_score = np.mean([r['partial_score'] for r in results])
        
        return {
            'accuracy': accuracy,
            'avg_partial_score': avg_partial_score,
            'total': len(results),
            'correct': sum(1 for r in results if r['correct']),
            'details': details
        }
    
    def run_few_shot_test(self, n_shots: int = 3, sample_size: int = 1319, batch_size: int = 16) -> Dict:
        """运行Few-Shot测试 - 批量优化版本"""
        print(f"开始Few-Shot测试 (n_shots={n_shots}, 批量大小: {batch_size})...")
        
        examples = self.dataset.get_few_shot_examples(n_shots)
        test_data = self.dataset.get_test_samples(sample_size)
        
        # 批量创建提示
        prompts = []
        for item in test_data:
            prompt = self.model.create_few_shot_prompt(item['question'], examples)
            prompts.append(prompt)
        
        # 批量生成响应
        print("正在批量生成响应...")
        responses = self.model.generate_batch_responses(prompts, batch_size=batch_size)
        
        # 处理结果
        results = []
        details = []
        
        print("正在处理结果...")
        for i, (item, response) in enumerate(tqdm(zip(test_data, responses), total=len(test_data))):
            question = item['question']
            ground_truth_answer = self.extract_answer(item['answer'])
            
            predicted_answer = self.extract_answer(response)
            # predicted_steps = self.extract_reasoning_steps(response)
            # partial_score = self.evaluate_partial_credit(predicted_steps, item['answer'])
            
            is_correct = (predicted_answer is not None and 
                         ground_truth_answer is not None and 
                         abs(predicted_answer - ground_truth_answer) < 0.01)
            
            result = {
                'correct': is_correct,
                # 'partial_score': partial_score
            }
            results.append(result)
            
            details.append({
                'question': question,
                'ground_truth': ground_truth_answer,
                'predicted': predicted_answer,
                'correct': is_correct,
                # 'partial_score': partial_score,
                'response': response
            })
        
        accuracy = self.calculate_accuracy(results)
        # avg_partial_score = np.mean([r['partial_score'] for r in results])
        
        return {
            'accuracy': accuracy,
            # 'avg_partial_score': avg_partial_score,
            'total': len(results),
            'correct': sum(1 for r in results if r['correct']),
            'details': details
        }
    
    def print_results(self, results: Dict, test_name: str):
        """打印测试结果"""
        print(f"\n{'='*50}")
        print(f"{test_name} 测试结果")
        print(f"{'='*50}")
        print(f"准确率: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
        # print(f"平均部分得分: {results['avg_partial_score']:.4f}")
        
        # 显示一些错误案例
        print(f"\n错误案例示例:")
        error_cases = [detail for detail in results['details'] if not detail['correct']]
        for i, case in enumerate(error_cases[:3]):
            print(f"\n案例 {i+1}:")
            print(f"问题: {case['question']}")
            print(f"正确答案: {case['ground_truth']}")
            print(f"预测答案: {case['predicted']}")
            # print(f"部分得分: {case['partial_score']:.2f}")
            print(f"模型回答: {case['response'][:200]}...")
    
    def run_comprehensive_test(self, sample_size: int = 1319, batch_size: int = 16):
        """运行全面测试 - 批量优化版本"""
        # Zero-Shot测试
        zero_shot_results = self.run_zero_shot_test(sample_size, batch_size)
        self.print_results(zero_shot_results, "Zero-Shot")
        #200个数据，batch_size=8，8分钟 50/200 25%
        #96个数据，batch_size=16, 2:12 28/96 29%
        #96个数据，batch_size=32, 1:09 36/96 37%
        #128个数据，batch_size=64, 0:47 40/128 31%
        #128个数据，batch_size=32, 1:38 36/128 28%
        
        # Few-Shot测试 (3-shot)
        few_shot_3_results = self.run_few_shot_test(n_shots=3, sample_size=sample_size, batch_size=batch_size)
        self.print_results(few_shot_3_results, "3-Shot")
        #128个数据，batch_size=32, 1:36 40/128 31%
        
        # Few-Shot测试 (5-shot)
        few_shot_5_results = self.run_few_shot_test(n_shots=5, sample_size=sample_size, batch_size=batch_size)
        self.print_results(few_shot_5_results, "5-Shot")
        #128个数据，batch_size=32, 1:39 42/128 33%
        
        # 汇总结果
        print(f"\n{'='*60}")
        print("汇总结果对比")
        print(f"{'='*60}")
        print(f"{'测试类型':<21} {'准确率':<16} {'部分得分':<10}")
        print(f"{'-'*60}")
        print(f"{'Zero-Shot':<25} {zero_shot_results['accuracy']:<20.4f} {zero_shot_results['avg_partial_score']:<20.4f}")
        print(f"{'3-Shot':<25} {few_shot_3_results['accuracy']:<20.4f} {few_shot_3_results['avg_partial_score']:<20.4f}")
        print(f"{'5-Shot':<25} {few_shot_5_results['accuracy']:<20.4f} {few_shot_5_results['avg_partial_score']:<20.4f}")
        
        return {
            'zero_shot': zero_shot_results,
            'few_shot_3': few_shot_3_results,
            'few_shot_5': few_shot_5_results
        }

# 使用示例
if __name__ == "__main__":
    # 初始化数据集
    dataset_name = "/root/autodl-tmp/GRPO_MATH/gsm8k"
    dataset = GSM8KDataset(dataset_name, "main")
    
    # 初始化模型
    model_path = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
    model = QwenModel(model_path)
    
    # 初始化测试器
    tester = Tester(dataset, model)
    
    # 运行全面测试，使用批量处理
    # 根据你的GPU显存调整batch_size，建议从8开始尝试，如果显存够用可以增加到16或32
    sample_size=1319
    results = tester.run_comprehensive_test(sample_size=sample_size, batch_size=32)


    # 保存结果
    with open("gsm8k_test_results_optimized.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到 gsm8k_test_results_optimized.json")