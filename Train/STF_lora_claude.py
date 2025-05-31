import os
import re
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
from tqdm import tqdm  # 添加tqdm导入
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, Dataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置类"""
    # 路径配置
    model_path: str = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
    dataset_path: str = "/root/autodl-tmp/GRPO_MATH/gsm8k"
    output_dir: str = "./lora_finetuned_qwen"
    
    # 训练参数
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    train_val_split_ratio: float = 0.8
    
    # LoRA参数
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # 其他配置
    use_bf16: bool = True
    logging_steps: int = 10
    
    def __post_init__(self):
        if self.target_modules is None:
            # self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            self.target_modules = ["q_proj", "v_proj"]

class GSM8KDataHandler:
    """GSM8K数据集处理器"""
    
    def __init__(self, config: TrainingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
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
        
    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """加载GSM8K数据集"""
        logger.info("开始加载GSM8K数据集...")
        
        try:
            # 尝试加载Parquet格式
            parquet_path = os.path.join(self.config.dataset_path, "main/train-00000-of-00001.parquet")
            if os.path.exists(parquet_path):
                logger.info(f"从Parquet文件加载: {parquet_path}")
                dataset = load_dataset("parquet", data_files={"train": parquet_path})
                train_data = list(dataset['train'])
            else:
                # 尝试其他格式
                logger.info("尝试从目录加载数据集...")
                dataset = load_dataset(self.config.dataset_path)
                train_data = list(dataset['train'])
                
        except Exception as e:
            logger.error(f"加载本地数据集失败: {e}")
            logger.info("尝试从HuggingFace加载GSM8K...")
            dataset = load_dataset("gsm8k", "default")
            train_data = list(dataset['train'])
        
        logger.info(f"原始训练集大小: {len(train_data)}")
        
        # 数据预处理
        processed_data = self._preprocess_data(train_data)
        
        # 创建Dataset对象并划分
        full_dataset = Dataset.from_list(processed_data)
        dataset_split = full_dataset.train_test_split(
            test_size=1 - self.config.train_val_split_ratio, 
            seed=42
        )
        
        train_dataset = dataset_split['train']
        val_dataset = dataset_split['test']
        
        logger.info(f"划分后训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def load_test_dataset(self) -> Dataset:
        """加载GSM8K测试集"""
        logger.info("开始加载GSM8K测试集...")
        
        try:
            # 加载测试集Parquet文件
            test_parquet_path = os.path.join(self.config.dataset_path, "main/test-00000-of-00001.parquet")
            if os.path.exists(test_parquet_path):
                logger.info(f"从Parquet文件加载测试集: {test_parquet_path}")
                dataset = load_dataset("parquet", data_files={"test": test_parquet_path})
                test_data = list(dataset['test'])
            else:
                # 尝试从HuggingFace加载
                logger.info("尝试从HuggingFace加载GSM8K测试集...")
                dataset = load_dataset("gsm8k", "default")
                test_data = list(dataset['test'])
                
        except Exception as e:
            logger.error(f"加载测试集失败: {e}")
            return None
        
        logger.info(f"测试集大小: {len(test_data)}")
        
        # 预处理测试数据
        processed_test_data = self._preprocess_data(test_data)
        test_dataset = Dataset.from_list(processed_test_data)
        
        return test_dataset
    
    def _preprocess_data(self, raw_data: List[Dict]) -> List[Dict]:
        """预处理原始数据，提取答案标签"""
        processed_data = []
        
        for sample in raw_data:
            # 提取最终答案数字
            answer_label = self.extract_answer_from_text(sample['answer'])
            
            # 构造输入文本，保持CoT格式
            input_text = f"问题: {sample['question']}\n回答: {sample['answer']}"
            
            processed_data.append({
                "text": input_text,
                "answer": sample['answer'],
                "answer_label": answer_label,
                "question": sample['question']
            })
        
        logger.info(f"预处理完成，样本数量: {len(processed_data)}")
        # # 打印几个样本的答案提取结果
        # for i in range(min(3, len(processed_data))):
        #     logger.info(f"样本{i+1} 原始答案: {processed_data[i]['answer'][:50]}...")
        #     logger.info(f"样本{i+1} 提取答案: {processed_data[i]['answer_label']}")
        
        return processed_data
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """对数据集进行tokenization"""
        logger.info("开始tokenization...")
        
        def tokenize_function(examples):
            # 编码完整文本（问题+回答）
            encodings = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # 编码答案部分作为labels
            answer_encodings = self.tokenizer(
                examples["answer"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # 设置labels，忽略padding部分
            encodings["labels"] = answer_encodings["input_ids"].clone()
            encodings["labels"][answer_encodings["attention_mask"] == 0] = -100
            
            return encodings
        
        # 应用tokenization
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            num_proc=1,
            desc="Tokenizing dataset"
        )
        
        # 移除原始文本列并设置格式
        tokenized_dataset = tokenized_dataset.remove_columns([col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]])
        tokenized_dataset.set_format("torch")
        
        return tokenized_dataset

class QwenLoRATrainer:
    """Qwen LoRA微调训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._setup_environment()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.original_model = None  # 保存原始模型的引用
        
    def _setup_environment(self):
        """设置环境变量"""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info(f"加载模型和分词器: {self.config.model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.original_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16
        )
        
        logger.info(f"模型参数量: {self.original_model.num_parameters():,}")
        
    def setup_lora(self):
        """配置LoRA"""
        logger.info("配置LoRA参数...")
        
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.original_model, lora_config)
        
        # 打印可训练参数信息
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """准备训练数据集"""
        data_handler = GSM8KDataHandler(self.config, self.tokenizer)
        
        # 加载原始数据集
        train_dataset, val_dataset = data_handler.load_dataset()
        
        # tokenization
        train_dataset = data_handler.tokenize_dataset(train_dataset)
        val_dataset = data_handler.tokenize_dataset(val_dataset)
        
        return train_dataset, val_dataset
        
    def setup_trainer(self, train_dataset: Dataset, val_dataset: Dataset):
        """设置训练器"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            logging_steps=self.config.logging_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=not self.config.use_bf16,
            bf16=self.config.use_bf16,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
    
    def load_finetuned_model(self):
        """加载微调后的模型进行推理"""
        logger.info("加载微调后的模型进行推理...")
        
        # 如果已经有训练好的模型，使用它
        if self.model is not None:
            logger.info("使用当前训练好的模型")
            return
        
        # 否则从保存的路径加载
        if os.path.exists(self.config.output_dir):
            logger.info(f"从 {self.config.output_dir} 加载微调后的模型")
            
            # 重新加载基础模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True, padding_side='left')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16
            )
            
            # 加载LoRA适配器
            self.model = PeftModel.from_pretrained(base_model, self.config.output_dir)
            logger.info("微调后的模型加载完成")
        else:
            raise ValueError(f"找不到微调后的模型路径: {self.config.output_dir}")
        
    def train(self):
        """执行完整的训练流程"""
        logger.info("开始训练流程...")
        
        try:
            # 1. 加载模型和分词器
            self.load_model_and_tokenizer()
            
            # 2. 配置LoRA
            self.setup_lora()
            
            # 3. 准备数据集
            train_dataset, val_dataset = self.prepare_datasets()
            
            # 4. 设置训练器
            self.setup_trainer(train_dataset, val_dataset)
            
            # 5. 清理显存
            torch.cuda.empty_cache()
            
            # 6. 开始训练
            logger.info("开始训练...")
            self.trainer.train()
            
            # 7. 保存模型
            logger.info("保存模型...")
            self.model.save_pretrained(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            logger.info("训练完成!")
            
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise
            
    def generate_answer(self, question: str) -> Dict[str, Union[str, Optional[float]]]:
        """生成答案"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型和分词器未加载，请先运行训练或加载微调后的模型")
            
        input_text = f"问题: {question}\n回答: "
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_length=self.config.max_length,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取最终答案 - 使用修改后的函数
        data_handler = GSM8KDataHandler(self.config, self.tokenizer)
        final_answer = data_handler.extract_answer_from_text(generated_text)
        
        return {
            "generated_text": generated_text,
            "final_answer": final_answer
        }
    
    def evaluate_on_test_set(self):
        """在测试集上评估模型性能"""
        logger.info("开始在GSM8K测试集上评估...")
        
        # 确保使用微调后的模型
        self.load_finetuned_model()
        
        # 加载测试数据
        data_handler = GSM8KDataHandler(self.config, self.tokenizer)
        test_dataset = data_handler.load_test_dataset()
        
        if test_dataset is None:
            logger.error("无法加载测试集")
            return
        
        self.model.eval()
        correct = 0
        total = 0
        
        # 取前100个样本进行快速评估（可以根据需要调整）
        test_samples = test_dataset.select(range(min(100, len(test_dataset))))
        
        # 使用tqdm显示进度
        progress_bar = tqdm(test_samples, desc="评估测试集", unit="样本")
        
        for sample in progress_bar:
            total += 1
            question = sample['question']
            true_answer = sample['answer_label']
            
            try:
                result = self.generate_answer(question)
                predicted_answer = result['final_answer']
                
                # 比较答案 - 现在比较的是浮点数
                if predicted_answer is not None and true_answer is not None:
                    # 使用浮点数比较，允许小的误差
                    if abs(predicted_answer - true_answer) < 1e-6:
                        correct += 1
                elif predicted_answer == true_answer:  # 都是None的情况
                    correct += 1
                
                # 更新进度条显示当前准确率
                current_accuracy = correct / total if total > 0 else 0
                progress_bar.set_postfix({
                    "准确率": f"{current_accuracy:.3f}",
                    "正确/总计": f"{correct}/{total}"
                })
                
                if total <= 5:  # 打印前5个样本的详细结果
                    logger.info(f"\n=== 测试样本 {total} ===")
                    logger.info(f"问题: {question}")
                    logger.info(f"真实答案: {true_answer}")
                    logger.info(f"预测答案: {predicted_answer}")
                    is_correct = (predicted_answer is not None and true_answer is not None and 
                                abs(predicted_answer - true_answer) < 1e-6) or (predicted_answer == true_answer)
                    logger.info(f"是否正确: {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                logger.error(f"评估样本 {total} 时出错: {e}")
        
        progress_bar.close()
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"\n=== 测试集评估结果 ===")
        logger.info(f"总样本数: {total}")
        logger.info(f"正确预测: {correct}")
        logger.info(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 保存评估结果
        eval_results = {
            "total_samples": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_config": {
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs
            }
        }
        
        eval_path = os.path.join(self.config.output_dir, "test_evaluation_results.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存到: {eval_path}")
        
        return eval_results

def save_config(config: TrainingConfig, filepath: str):
    """保存配置到文件"""
    config_dict = {
        'model_path': config.model_path,
        'dataset_path': config.dataset_path,
        'output_dir': config.output_dir,
        'max_length': config.max_length,
        'batch_size': config.batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'num_epochs': config.num_epochs,
        'learning_rate': config.learning_rate,
        'lora_rank': config.lora_rank,
        'lora_alpha': config.lora_alpha,
        'lora_dropout': config.lora_dropout,
        'target_modules': config.target_modules,
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def load_config(filepath: str) -> TrainingConfig:
    """从文件加载配置"""
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return TrainingConfig(**config_dict)

def main():
    """主函数"""
    # 创建训练配置
    config = TrainingConfig(
        model_path="/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B",
        dataset_path="/root/autodl-tmp/GRPO_MATH/gsm8k",
        output_dir="./lora_finetuned_qwen",
        max_length=512,
        batch_size=4,
        gradient_accumulation_steps=2,
        num_epochs=3,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    # 保存配置
    os.makedirs(config.output_dir, exist_ok=True)
    save_config(config, os.path.join(config.output_dir, "training_config.json"))
    
    try:
        # 创建训练器并执行训练
        trainer = QwenLoRATrainer(config)
        
        # 执行训练
        logger.info("=" * 50)
        logger.info("开始模型训练")
        logger.info("=" * 50)
        trainer.train()
        
        # 训练完成后进行测试集评估
        logger.info("=" * 50)
        logger.info("训练完成，开始测试集评估")
        logger.info("=" * 50)
        eval_results = trainer.evaluate_on_test_set()
        
        # 显示最终结果摘要
        logger.info("=" * 50)
        logger.info("训练和评估完成！最终结果摘要：")
        logger.info("=" * 50)
        logger.info(f"✅ 模型训练完成，保存路径: {config.output_dir}")
        logger.info(f"📊 测试集评估结果:")
        logger.info(f"   - 测试样本数: {eval_results['total_samples']}")
        logger.info(f"   - 正确预测数: {eval_results['correct_predictions']}")
        logger.info(f"   - 准确率: {eval_results['accuracy']:.3f} ({eval_results['accuracy']*100:.1f}%)")
        logger.info(f"📄 详细评估报告保存至: {os.path.join(config.output_dir, 'test_evaluation_results.json')}")
        
        
        
        logger.info("=" * 50)
        logger.info("所有任务完成！")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 即使训练失败，也尝试保存一些有用的信息
        error_info = {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "config": {
                "model_path": config.model_path,
                "dataset_path": config.dataset_path,
                "output_dir": config.output_dir,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout
            }
        }

        error_path = os.path.join(config.output_dir, "error_log.json")
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"错误信息已保存到: {error_path}")
    
    finally:
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理CUDA缓存")

if __name__ == "__main__":
    main()