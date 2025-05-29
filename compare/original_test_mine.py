import re
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from qwen_model import QwenModel
from gsm8k_dataset import GSM8KDataset
import json
import time


class Tester:
    """测试类"""
    
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
        # 按句号或换行符分割
        steps = re.split(r'[.\n]', text)
        # 过滤掉空步骤和太短的步骤
        steps = [step.strip() for step in steps if len(step.strip()) > 10]
        return steps
    
    def calculate_accuracy(self, results: List[Dict]) -> float:
        """
        计算准确率
        
        Args:
            results: 测试结果列表，每个元素包含 'correct' 字段
            
        Returns:
            准确率 (0-1之间的浮点数)
        """
        if not results:
            return 0.0
        
        correct_count = sum(1 for result in results if result.get('correct', False))
        total_count = len(results)
        
        return correct_count / total_count
    
    def evaluate_partial_credit(self, predicted_steps: List[str], ground_truth: str) -> float:
        """评估部分得分 - 基于推理步骤的正确性"""
        if not predicted_steps:
            return 0.0
        
        # 从标准答案中提取数学表达式
        gt_expressions = re.findall(r'<<([^>]+)>>', ground_truth)
        
        correct_steps = 0
        total_steps = len(predicted_steps)
        
        for step in predicted_steps:
            # 检查步骤中是否包含正确的数学计算
            step_expressions = re.findall(r'(\d+(?:\.\d+)?)\s*[+\-*/]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', step)
            
            for expr in step_expressions:
                try:
                    a, op_match = float(expr[0]), re.search(r'[+\-*/]', step)
                    if op_match:
                        op = op_match.group()
                        b, result = float(expr[1]), float(expr[2])
                        
                        # 验证计算是否正确
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
                # 如果没有数学表达式，检查是否包含合理的推理
                if any(keyword in step.lower() for keyword in ['because', 'so', 'therefore', 'thus', 'since']):
                    correct_steps += 0.5
        
        return correct_steps / total_steps if total_steps > 0 else 0.0
    
    def run_zero_shot_test(self, sample_size: int = 1319) -> Dict:
        """运行Zero-Shot测试"""
        print("开始Zero-Shot测试...")
        test_data = self.dataset.get_test_samples(sample_size)
        
        results = []
        details = []
        
        for item in tqdm(test_data, desc="Zero-Shot测试"):
            start = time.time()
            question = item['question']
            ground_truth_answer = self.extract_answer(item['answer'])
            print(f"Time: {(time.time() - start):.3f}s")
            
            prompt = self.model.create_zero_shot_prompt(question)
            response = self.model.generate_response(prompt)
            
            print(f"Time: {(time.time() - start):.3f}s")
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
            print(f"Time: {(time.time() - start):.3f}s")
        
        accuracy = self.calculate_accuracy(results)
        avg_partial_score = np.mean([r['partial_score'] for r in results])
        
        return {
            'accuracy': accuracy,
            'avg_partial_score': avg_partial_score,
            'total': len(results),
            'correct': sum(1 for r in results if r['correct']),
            'details': details
        }
    
    def run_few_shot_test(self, n_shots: int = 3, sample_size: int = 1319) -> Dict:
        """运行Few-Shot测试"""
        print(f"开始Few-Shot测试 (n_shots={n_shots})...")
        
        # 获取示例和测试数据
        examples = self.dataset.get_few_shot_examples(n_shots)
        test_data = self.dataset.get_test_samples(sample_size)
        
        results = []
        details = []
        
        for item in tqdm(test_data, desc=f"{n_shots}-shot测试"):
            question = item['question']
            ground_truth_answer = self.extract_answer(item['answer'])
            
            prompt = self.model.create_few_shot_prompt(question, examples)
            response = self.model.generate_response(prompt)

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
    
    def print_results(self, results: Dict, test_name: str):
        """打印测试结果"""
        print(f"\n{'='*50}")
        print(f"{test_name} 测试结果")
        print(f"{'='*50}")
        print(f"准确率: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
        print(f"平均部分得分: {results['avg_partial_score']:.4f}")
        
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
        """运行全面测试"""
        # Zero-Shot测试
        zero_shot_results = self.run_zero_shot_test(sample_size)
        self.print_results(zero_shot_results, "Zero-Shot")
        
        # Few-Shot测试 (3-shot)
        few_shot_3_results = self.run_few_shot_test(n_shots=3, sample_size=sample_size)
        self.print_results(few_shot_3_results, "3-Shot")
        
        # Few-Shot测试 (5-shot)
        few_shot_5_results = self.run_few_shot_test(n_shots=5, sample_size=sample_size)
        self.print_results(few_shot_5_results, "5-Shot")
        
        # 汇总结果
        print(f"\n{'='*60}")
        print("汇总结果对比")
        print(f"{'='*60}")
        print(f"{'测试类型':<20} {'准确率':<10} {'部分得分':<10}")
        print(f"{'-'*40}")
        print(f"{'Zero-Shot':<20} {zero_shot_results['accuracy']:<10.4f} {zero_shot_results['avg_partial_score']:<10.4f}")
        print(f"{'3-Shot':<20} {few_shot_3_results['accuracy']:<10.4f} {few_shot_3_results['avg_partial_score']:<10.4f}")
        print(f"{'5-Shot':<20} {few_shot_5_results['accuracy']:<10.4f} {few_shot_5_results['avg_partial_score']:<10.4f}")
        
        return {
            'zero_shot': zero_shot_results,
            'few_shot_3': few_shot_3_results,
            'few_shot_5': few_shot_5_results
        }

# 使用示例
if __name__ == "__main__":
    # 初始化数据集
    dataset_name = "/root/autodl-tmp/GRPO_MATH/gsm8k"
    dataset = GSM8KDataset(dataset_name, "main")  # 也可以使用 "socratic"
    
    # 初始化模型
    model_path = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
    model = QwenModel(model_path)
    
    # 初始化测试器
    tester = Tester(dataset, model)
    
    # 运行全面测试
    results = tester.run_comprehensive_test(sample_size=20)
    
    # 保存结果
    with open("gsm8k_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到 gsm8k_test_results.json")
    
    # 单独测试准确率计算
    print("\n测试准确率计算函数:")
    test_results = [
        {'correct': True}, {'correct': False}, {'correct': True}, 
        {'correct': True}, {'correct': False}
    ]
    accuracy = tester.calculate_accuracy(test_results)
    print(f"测试数据准确率: {accuracy:.4f} (应该是 0.6000)")