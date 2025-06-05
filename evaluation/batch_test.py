import os
import re
import sys
import json
import torch
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset, Dataset

# 添加项目根目录到Python路径，便于导入其他模块
sys.path.append("/root/autodl-tmp/GRPO_MATH")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """批量测试配置类"""
    # 路径配置
    base_model_path: str = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
    model_path: str = "./lora_finetuned_qwen"  # 可以是LoRA或GRPO的模型路径
    dataset_path: str = "/root/autodl-tmp/GRPO_MATH/gsm8k"
    output_dir: str = "./evaluation_results"
    
    # 测试参数
    max_length: int = 512
    sample_size: int = 100
    batch_size: int = 8
    temperature: float = 0.1
    max_new_tokens: int = 256
    
    # 评估模式
    model_type: str = "lora"  # 'lora' 或 'grpo'
    
    # 其他配置
    use_bf16: bool = True
    
    def __post_init__(self):
        """后处理配置"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)


class GSM8KDataHandler:
    """GSM8K数据集处理器 - 从原文件复用"""
    
    def __init__(self, config: TestConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
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
        
    def load_test_data(self) -> List[Dict]:
        """加载GSM8K测试数据"""
        logger.info("开始加载GSM8K测试数据...")
        
        try:
            # 加载Parquet格式
            test_parquet_path = os.path.join(self.config.dataset_path, "main/test-00000-of-00001.parquet")
            
            if os.path.exists(test_parquet_path):
                logger.info(f"从Parquet文件加载测试数据")
                dataset = load_dataset("parquet", data_files={"test": test_parquet_path})
                self.test_data = list(dataset['test'])
            else:
                raise FileNotFoundError(f"找不到测试数据文件: {test_parquet_path}")
                
        except Exception as e:
            logger.error(f"加载测试数据失败: {e}")
            raise
        
        logger.info(f"测试集大小: {len(self.test_data)}")
        return self.test_data
    
    def get_test_samples(self, n_samples: Optional[int] = None) -> List[Dict]:
        """获取测试样本"""
        if self.test_data is None:
            self.load_test_data()
            
        if n_samples is None:
            return self.test_data
        return self.test_data[:min(n_samples, len(self.test_data))]
    
    def create_prompts(self, test_samples: List[Dict]) -> List[str]:
        """为测试样本创建提示"""
        prompts = []
        
        for sample in test_samples:
            # 尝试使用chat template格式化
            try:
                conversation = [
                    {"role": "user", "content": "Please solve this math problem step by step and provide your final answer after the \"####\" marker.\n"+sample['question']},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"应用chat template失败: {e}，使用简单格式")
                # 如果失败，使用简单格式
                prompt = f"Please solve this math problem step by step and provide your final answer after the \"####\" marker.\nQuestion: {sample['question']}\nAnswer: "
            
            prompts.append(prompt)
        
        return prompts


class ModelTester:
    """模型批量测试器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.data_handler = None
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info(f"加载基础模型: {self.config.base_model_path}")
        logger.info(f"加载微调模型: {self.config.model_path}")
        
        # 先尝试从微调模型目录加载分词器
        try:
            logger.info(f"尝试从微调模型目录加载分词器")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, 
                trust_remote_code=True, 
                padding_side='left'
            )
            logger.info("成功从微调模型目录加载分词器")
        except Exception as e:
            logger.warning(f"从微调模型目录加载分词器失败: {e}")
            # 如果失败，从基础模型加载
            logger.info(f"从基础模型加载分词器")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path, 
                trust_remote_code=True, 
                padding_side='left'
            )
        
        # 确保有padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        logger.info(f"加载基础模型: {self.config.base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16
        )
        
        # 加载微调模型
        try:
            logger.info(f"加载微调适配器: {self.config.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
            logger.info("微调模型加载成功")
        except Exception as e:
            logger.error(f"加载微调模型失败: {e}")
            logger.warning("使用未微调的基础模型继续")
            self.model = base_model
        
        # 设置为评估模式
        self.model.eval()
    
    def generate_single_response(self, prompt: str) -> str:
        """生成单个响应，用于详细展示"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=False,  # 评估用贪婪解码
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def generate_batch_responses(self, prompts: List[str]) -> List[str]:
        """批量生成模型响应"""
        all_responses = []
        
        # 分批处理
        for i in tqdm(range(0, len(prompts), self.config.batch_size), desc="批量生成"):
            batch_prompts = prompts[i:i + self.config.batch_size]
            
            # 批量编码
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.max_length // 2  # 预留一半长度给生成
            ).to(self.model.device)
            
            with torch.no_grad():
                # 批量生成
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=False,  # 评估用贪婪解码
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
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
    
    def run_batch_test(self):
        """运行批量测试"""
        logger.info(f"开始批量测试 (样本数: {self.config.sample_size}, 批量大小: {self.config.batch_size})...")
        
        # 加载模型和分词器
        self.load_model_and_tokenizer()
        
        # 初始化数据处理器
        self.data_handler = GSM8KDataHandler(self.config, self.tokenizer)
        
        # 获取测试样本
        test_samples = self.data_handler.get_test_samples(self.config.sample_size)
        logger.info(f"获取测试样本数量: {len(test_samples)}")
        
        # 创建提示
        prompts = self.data_handler.create_prompts(test_samples)
        
        # 处理结果
        results = []
        details = []
        correct = 0
        format_correct = 0
        
        logger.info("=" * 80)
        logger.info("前3个测试样本的详细推理过程:")
        logger.info("=" * 80)

        # 先处理前3个样本，展示详细推理过程
        for i in range(min(3, len(test_samples))):
            sample = test_samples[i]
            prompt = prompts[i]
            
            logger.info(f"\n{'='*20} 测试样本 {i+1} {'='*20}")
            logger.info(f"问题: {sample['question']}")
            logger.info(f"标准答案: {sample['answer']}")
            logger.info(f"标准答案数值: {self.data_handler.extract_answer_from_text(sample['answer'])}")
            logger.info(f"输入提示:\n{prompt}")
            
            # 单独生成这个样本的响应，展示详细过程
            response = self.generate_single_response(prompt)
            
            logger.info(f"模型完整响应:\n{response}")
            
            predicted_answer = self.data_handler.extract_answer_from_text(response)
            ground_truth_answer = self.data_handler.extract_answer_from_text(sample['answer'])
            
            # 检查格式是否正确
            has_correct_format = self.data_handler.has_correct_format(response)
            if has_correct_format:
                format_correct += 1
            
            # 比较答案
            is_correct = (predicted_answer is not None and 
                         ground_truth_answer is not None and 
                         abs(predicted_answer - ground_truth_answer) < 1e-6)
            
            if is_correct:
                correct += 1
            
            logger.info(f"提取的预测答案: {predicted_answer}")
            logger.info(f"提取的标准答案: {ground_truth_answer}")
            logger.info(f"答案正确: {'✅' if is_correct else '❌'}")
            logger.info(f"格式正确: {'✅' if has_correct_format else '❌'}")
            
            results.append({'correct': is_correct, 'format_correct': has_correct_format})
            details.append({
                'question': sample['question'],
                'ground_truth': ground_truth_answer,
                'predicted': predicted_answer,
                'correct': is_correct,
                'format_correct': has_correct_format,
                'response': response,
                'prompt': prompt
            })
        
        logger.info("\n" + "=" * 80)
        logger.info("开始批量处理剩余样本...")
        logger.info("=" * 80)
        
        # 批量处理剩余样本
        if len(test_samples) > 3:
            remaining_prompts = prompts[3:]
            remaining_samples = test_samples[3:]
            
            logger.info("正在批量生成剩余响应...")
            remaining_responses = self.generate_batch_responses(remaining_prompts)
            
            logger.info("正在处理剩余结果...")
            for i, (sample, response) in enumerate(tqdm(zip(remaining_samples, remaining_responses), desc="处理剩余结果")):
                question = sample['question']
                ground_truth_text = sample['answer']
                ground_truth_answer = self.data_handler.extract_answer_from_text(ground_truth_text)
                
                predicted_answer = self.data_handler.extract_answer_from_text(response)
                
                # 检查格式是否正确
                has_correct_format = self.data_handler.has_correct_format(response)
                if has_correct_format:
                    format_correct += 1
                
                # 比较答案
                is_correct = (predicted_answer is not None and 
                             ground_truth_answer is not None and 
                             abs(predicted_answer - ground_truth_answer) < 1e-6)
                
                if is_correct:
                    correct += 1
                
                results.append({'correct': is_correct, 'format_correct': has_correct_format})
                details.append({
                    'question': question,
                    'ground_truth': ground_truth_answer,
                    'predicted': predicted_answer,
                    'correct': is_correct,
                    'format_correct': has_correct_format,
                    'response': response,
                    'prompt': prompts[3 + i]
                })
        
        # 计算评估指标
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        format_accuracy = format_correct / total if total > 0 else 0
        
        # 显示评估结果
        logger.info(f"\n" + "=" * 80)
        logger.info("批量测试最终结果")
        logger.info("=" * 80)
        logger.info(f"总样本数: {total}")
        logger.info(f"答案正确数: {correct}")
        logger.info(f"格式正确数: {format_correct}")
        logger.info(f"答案准确率: {accuracy:.4f} ({accuracy*100:.1f}%)")
        logger.info(f"格式准确率: {format_accuracy:.4f} ({format_accuracy*100:.1f}%)")
        
        # 保存评估结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(self.config.model_path.rstrip("/"))
        eval_results = {
            "model_path": self.config.model_path,
            "model_type": self.config.model_type,
            "total_samples": total,
            "correct_predictions": correct,
            "format_correct": format_correct,
            "answer_accuracy": accuracy,
            "format_accuracy": format_accuracy,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "sample_size": self.config.sample_size,
                "batch_size": self.config.batch_size,
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_new_tokens
            },
            "details": details
        }
        
        eval_path = os.path.join(self.config.output_dir, f"{model_name}_{timestamp}_test_results.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"详细评估结果已保存到: {eval_path}")
        
        return eval_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量测试GSM8K数据集上的微调模型')
    
    parser.add_argument('--base_model', type=str, default="/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B",
                        help='基础模型路径')
    parser.add_argument('--model', type=str, required=True,
                        help='微调模型路径')
    parser.add_argument('--dataset', type=str, default="/root/autodl-tmp/GRPO_MATH/gsm8k",
                        help='GSM8K数据集路径')
    parser.add_argument('--output', type=str, default="./evaluation_results",
                        help='评估结果输出目录')
    parser.add_argument('--model_type', type=str, choices=['lora', 'grpo'], default='lora',
                        help='模型类型 (lora 或 grpo)')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='测试样本数量')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批处理大小')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='生成温度')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='最大生成token数量')
    parser.add_argument('--no_bf16', action='store_true',
                        help='不使用bf16')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建测试配置
    config = TestConfig(
        base_model_path=args.base_model,
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        model_type=args.model_type,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_bf16=not args.no_bf16,
    )
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 记录配置
    config_path = os.path.join(config.output_dir, f"test_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        # 转换为字典并保存
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    try:
        # 创建测试器并运行测试
        tester = ModelTester(config)
        
        logger.info("=" * 50)
        logger.info(f"开始对模型 {config.model_path} 进行批量测试")
        logger.info("=" * 50)
        
        # 运行测试
        eval_results = tester.run_batch_test()
        
        # 显示最终结果摘要
        logger.info("=" * 50)
        logger.info("批量测试完成！最终结果摘要：")
        logger.info("=" * 50)
        logger.info(f"📊 测试结果:")
        logger.info(f"   - 测试样本数: {eval_results['total_samples']}")
        logger.info(f"   - 答案正确数: {eval_results['correct_predictions']}")
        logger.info(f"   - 答案准确率: {eval_results['answer_accuracy']:.4f} ({eval_results['answer_accuracy']*100:.1f}%)")
        logger.info(f"   - 格式正确数: {eval_results['format_correct']}")
        logger.info(f"   - 格式准确率: {eval_results['format_accuracy']:.4f} ({eval_results['format_accuracy']*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理CUDA缓存")


if __name__ == "__main__":
    main()
