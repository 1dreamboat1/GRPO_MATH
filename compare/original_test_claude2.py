import json
import re
import random
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time

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

class ThreadSafeQwenModel:
    """线程安全的Qwen模型类"""
    
    def __init__(self, model_path: str = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B", num_threads: int = 4):
        """
        初始化模型
        
        Args:
            model_path: 模型路径
            num_threads: 并行线程数
        """
        self.model_path = model_path
        self.num_threads = num_threads
        self.tokenizer = None
        self.model = None
        self.device = None
        self._model_lock = threading.Lock()  # 模型访问锁
        self._load_model()
        
    def _load_model(self):
        """加载模型和分词器"""
        print("正在加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
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
        
    def generate_response_safe(self, prompt: str, max_length: int = 512, temperature: float = 0.1) -> str:
        """
        线程安全的单个响应生成
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
        """
        with self._model_lock:  # 确保模型访问的线程安全
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除输入提示部分
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # 清理显存
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return response
    
    def generate_responses_parallel(self, prompts: List[str], max_length: int = 512, 
                                  temperature: float = 0.1) -> List[str]:
        """
        并行生成响应
        
        Args:
            prompts: 输入提示列表
            max_length: 最大生成长度
            temperature: 温度参数
        """
        responses = [None] * len(prompts)
        
        def process_prompt(idx_prompt_pair):
            idx, prompt = idx_prompt_pair
            try:
                response = self.generate_response_safe(prompt, max_length, temperature)
                return idx, response
            except Exception as e:
                print(f"处理提示 {idx} 时出错: {e}")
                return idx, ""
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(process_prompt, (i, prompt)): i 
                for i, prompt in enumerate(prompts)
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_idx), total=len(prompts), desc="并行生成响应"):
                idx, response = future.result()
                responses[idx] = response
        
        return responses
    
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

class MultiThreadTester:
    """多线程测试类"""
    
    def __init__(self, dataset: GSM8KDataset, model: ThreadSafeQwenModel):
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
    
    def process_results_parallel(self, test_data: List[Dict], responses: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """并行处理结果"""
        results = []
        details = []
        
        def process_single_result(idx_data_response):
            idx, (item, response) = idx_data_response
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
            
            detail = {
                'question': question,
                'ground_truth': ground_truth_answer,
                'predicted': predicted_answer,
                'correct': is_correct,
                'partial_score': partial_score,
                'response': response
            }
            
            return idx, result, detail
        
        # 并行处理结果
        indexed_results = [None] * len(test_data)
        indexed_details = [None] * len(test_data)
        
        with ThreadPoolExecutor(max_workers=self.model.num_threads) as executor:
            future_to_idx = {
                executor.submit(process_single_result, (i, (item, response))): i 
                for i, (item, response) in enumerate(zip(test_data, responses))
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(test_data), desc="处理结果"):
                idx, result, detail = future.result()
                indexed_results[idx] = result
                indexed_details[idx] = detail
        
        return indexed_results, indexed_details
    
    def run_zero_shot_test(self, sample_size: int = 1319) -> Dict:
        """运行Zero-Shot测试 - 多线程版本"""
        print(f"开始Zero-Shot测试 (并行线程数: {self.model.num_threads})...")
        test_data = self.dataset.get_test_samples(sample_size)
        
        # 创建提示
        print("创建提示...")
        prompts = []
        for item in test_data:
            prompt = self.model.create_zero_shot_prompt(item['question'])
            prompts.append(prompt)
        
        # 并行生成响应
        print("并行生成响应...")
        start_time = time.time()
        responses = self.model.generate_responses_parallel(prompts)
        generation_time = time.time() - start_time
        print(f"生成完成，耗时: {generation_time:.2f}秒")
        
        # 并行处理结果
        print("并行处理结果...")
        results, details = self.process_results_parallel(test_data, responses)
        
        accuracy = self.calculate_accuracy(results)
        avg_partial_score = np.mean([r['partial_score'] for r in results])
        
        return {
            'accuracy': accuracy,
            'avg_partial_score': avg_partial_score,
            'total': len(results),
            'correct': sum(1 for r in results if r['correct']),
            'details': details,
            'generation_time': generation_time
        }
    
    def run_few_shot_test(self, n_shots: int = 3, sample_size: int = 1319) -> Dict:
        """运行Few-Shot测试 - 多线程版本"""
        print(f"开始Few-Shot测试 (n_shots={n_shots}, 并行线程数: {self.model.num_threads})...")
        
        examples = self.dataset.get_few_shot_examples(n_shots)
        test_data = self.dataset.get_test_samples(sample_size)
        
        # 创建提示
        print("创建提示...")
        prompts = []
        for item in test_data:
            prompt = self.model.create_few_shot_prompt(item['question'], examples)
            prompts.append(prompt)
        
        # 并行生成响应
        print("并行生成响应...")
        start_time = time.time()
        responses = self.model.generate_responses_parallel(prompts)
        generation_time = time.time() - start_time
        print(f"生成完成，耗时: {generation_time:.2f}秒")
        
        # 并行处理结果
        print("并行处理结果...")
        results, details = self.process_results_parallel(test_data, responses)
        
        accuracy = self.calculate_accuracy(results)
        avg_partial_score = np.mean([r['partial_score'] for r in results])
        
        return {
            'accuracy': accuracy,
            'avg_partial_score': avg_partial_score,
            'total': len(results),
            'correct': sum(1 for r in results if r['correct']),
            'details': details,
            'generation_time': generation_time
        }
    
    def print_results(self, results: Dict, test_name: str):
        """打印测试结果"""
        print(f"\n{'='*50}")
        print(f"{test_name} 测试结果")
        print(f"{'='*50}")
        print(f"准确率: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
        print(f"平均部分得分: {results['avg_partial_score']:.4f}")
        print(f"生成耗时: {results['generation_time']:.2f}秒")
        print(f"平均每题耗时: {results['generation_time']/results['total']:.2f}秒")
        
        # 显示一些错误案例
        print(f"\n错误案例示例:")
        error_cases = [detail for detail in results['details'] if not detail['correct']]
        for i, case in enumerate(error_cases[:3]):
            print(f"\n案例 {i+1}:")
            print(f"问题: {case['question']}")
            print(f"正确答案: {case['ground_truth']}")
            print(f"预测答案: {case['predicted']}")
            print(f"部分得分: {case['partial_score']:.2f}")
            print(f"模型回答: {case['response'][:200]}...")
    
    def run_comprehensive_test(self, sample_size: int = 1319):
        """运行全面测试 - 多线程版本"""
        total_start_time = time.time()
        
        # Zero-Shot测试
        zero_shot_results = self.run_zero_shot_test(sample_size)
        self.print_results(zero_shot_results, "Zero-Shot")
        
        # Few-Shot测试 (3-shot)
        few_shot_3_results = self.run_few_shot_test(n_shots=3, sample_size=sample_size)
        self.print_results(few_shot_3_results, "3-Shot")
        
        # Few-Shot测试 (5-shot)
        few_shot_5_results = self.run_few_shot_test(n_shots=5, sample_size=sample_size)
        self.print_results(few_shot_5_results, "5-Shot")
        
        total_time = time.time() - total_start_time
        
        # 汇总结果
        print(f"\n{'='*60}")
        print("汇总结果对比")
        print(f"{'='*60}")
        print(f"{'测试类型':<20} {'准确率':<10} {'部分得分':<10} {'耗时(秒)':<10}")
        print(f"{'-'*50}")
        print(f"{'Zero-Shot':<20} {zero_shot_results['accuracy']:<10.4f} {zero_shot_results['avg_partial_score']:<10.4f} {zero_shot_results['generation_time']:<10.2f}")
        print(f"{'3-Shot':<20} {few_shot_3_results['accuracy']:<10.4f} {few_shot_3_results['avg_partial_score']:<10.4f} {few_shot_3_results['generation_time']:<10.2f}")
        print(f"{'5-Shot':<20} {few_shot_5_results['accuracy']:<10.4f} {few_shot_5_results['avg_partial_score']:<10.4f} {few_shot_5_results['generation_time']:<10.2f}")
        print(f"\n总耗时: {total_time:.2f}秒")
        
        return {
            'zero_shot': zero_shot_results,
            'few_shot_3': few_shot_3_results,
            'few_shot_5': few_shot_5_results,
            'total_time': total_time
        }

# 使用示例
if __name__ == "__main__":
    # 初始化数据集
    dataset_name = "/root/autodl-tmp/GRPO_MATH/gsm8k"
    dataset = GSM8KDataset(dataset_name, "main")
    
    # 初始化模型 - 设置并行线程数
    # 建议线程数设置为2-4，避免过多线程导致显存不足
    model_path = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
    model = ThreadSafeQwenModel(model_path, num_threads=4)
    
    # 初始化测试器
    tester = MultiThreadTester(dataset, model)
    
    # 运行全面测试
    results = tester.run_comprehensive_test(sample_size=1319)
    
    # 保存结果
    with open("gsm8k_test_results_multithread.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items() if k != 'details'}, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到 gsm8k_test_results_multithread.json")
    
    # 性能报告
    print(f"\n{'='*60}")
    print("性能报告")
    print(f"{'='*60}")
    print(f"使用线程数: {model.num_threads}")
    print(f"测试样本数: {results['zero_shot']['total']}")
    print(f"总耗时: {results['total_time']:.2f}秒")
    print(f"平均每题耗时: {results['total_time']/(results['zero_shot']['total']*3):.2f}秒")
    print("注意: 以上时间包含了3种测试方法的总时间")