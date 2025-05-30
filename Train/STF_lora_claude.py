import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GSM8KDataProcessor:
    """GSM8K数据集处理器"""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def format_prompt(self, question, answer=None):
        """格式化提示词，保持CoT结构"""
        if answer is not None:
            # 训练时的格式
            prompt = f"问题：{question}\n\n解答：{answer}"
        else:
            # 推理时的格式
            prompt = f"问题：{question}\n\n解答："
        return prompt
    
    def preprocess_function(self, examples):
        """数据预处理函数"""
        texts = []
        for question, answer in zip(examples['question'], examples['answer']):
            formatted_text = self.format_prompt(question, answer)
            texts.append(formatted_text)
        
        # tokenization
        model_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        # 设置labels为input_ids的副本（用于语言建模）
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

class QwenLoRATrainer:
    """Qwen LoRA微调训练器"""
    
    def __init__(self, model_path, dataset_path):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info(f"加载模型和分词器从: {self.model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"模型参数量: {self.model.num_parameters():,}")
        
    def setup_lora(self):
        """配置LoRA参数"""
        logger.info("配置LoRA参数...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,                    # LoRA秩，控制适应的复杂度
            lora_alpha=32,           # LoRA缩放参数
            lora_dropout=0.1,        # LoRA层的dropout
            target_modules=[         # 目标模块，Qwen模型的注意力层
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            bias="none",             # 不训练bias
        )
        
        # 应用LoRA到模型
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
    def load_dataset(self):
        """加载和预处理GSM8K数据集"""
        logger.info(f"加载数据集从: {self.dataset_path}")
        
        # 检查是否为本地JSON文件
        if os.path.isfile(os.path.join(self.dataset_path, "train.jsonl")):
            # 加载本地JSONL文件
            train_data = []
            with open(os.path.join(self.dataset_path, "train.jsonl"), 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line.strip()))
            
            test_data = []
            test_file = os.path.join(self.dataset_path, "test.jsonl")
            if os.path.exists(test_file):
                with open(test_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        test_data.append(json.loads(line.strip()))
            
            # 转换为Dataset格式
            train_dataset = Dataset.from_list(train_data)
            test_dataset = Dataset.from_list(test_data) if test_data else None
            
        else:
            # 从HuggingFace加载
            dataset = load_dataset("gsm8k", "main")
            train_dataset = dataset["train"]
            test_dataset = dataset["test"]
        
        # 数据预处理
        data_processor = GSM8KDataProcessor(self.tokenizer)
        
        # 处理训练集
        train_dataset = train_dataset.map(
            data_processor.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="处理训练数据"
        )
        
        # 处理测试集（如果存在）
        if test_dataset:
            test_dataset = test_dataset.map(
                data_processor.preprocess_function,
                batched=True,
                remove_columns=test_dataset.column_names,
                desc="处理测试数据"
            )
        
        logger.info(f"训练样本数: {len(train_dataset)}")
        if test_dataset:
            logger.info(f"测试样本数: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def setup_training_arguments(self, output_dir="./qwen_lora_gsm8k"):
        """设置训练参数"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,              # 训练轮数
            per_device_train_batch_size=4,   # 训练批次大小
            per_device_eval_batch_size=4,    # 评估批次大小
            gradient_accumulation_steps=4,    # 梯度累积步数
            warmup_steps=100,                # 预热步数
            learning_rate=2e-4,              # 学习率
            weight_decay=0.01,               # 权重衰减
            logging_steps=50,                # 日志记录间隔
            save_steps=500,                  # 模型保存间隔
            eval_steps=500,                  # 评估间隔
            save_total_limit=3,              # 最多保存的检查点数量
            load_best_model_at_end=True,     # 训练结束后加载最佳模型
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,                  # 不使用wandb等工具
            dataloader_pin_memory=False,
            fp16=True,                       # 使用半精度训练
            gradient_checkpointing=True,     # 使用梯度检查点节省显存
            remove_unused_columns=False,
        )
        return training_args
    
    def train(self):
        """执行训练"""
        logger.info("开始训练流程...")
        
        # 1. 加载模型和分词器
        self.load_model_and_tokenizer()
        
        # 2. 设置LoRA
        self.setup_lora()
        
        # 3. 加载数据集
        train_dataset, eval_dataset = self.load_dataset()
        
        # 4. 设置训练参数
        training_args = self.setup_training_arguments()
        
        # 5. 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # 不使用掩码语言建模
            pad_to_multiple_of=8
        )
        
        # 6. 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 7. 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 8. 保存模型
        logger.info("保存最终模型...")
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        logger.info("训练完成!")
        
        return trainer
    
    def test_inference(self, test_questions=None):
        """测试推理效果"""
        if test_questions is None:
            test_questions = [
                "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                "A restaurant serves 240 meals per day. If each meal costs $12, how much money does the restaurant make in a week?"
            ]
        
        logger.info("测试推理效果...")
        self.model.eval()
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n=== 测试样例 {i} ===")
            logger.info(f"问题: {question}")
            
            # 格式化输入
            data_processor = GSM8KDataProcessor(self.tokenizer)
            prompt = data_processor.format_prompt(question)
            
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_start = generated_text.find("解答：") + 3
            answer = generated_text[answer_start:].strip()
            
            logger.info(f"生成答案: {answer}")

def main():
    """主函数"""
    # 配置路径
    model_path = "/root/autodl-tmp/GRPO_MATH/Qwen2_0.5B"
    dataset_path = "/root/autodl-tmp/GRPO_MATH/gsm8k"
    
    # 创建训练器并开始训练
    trainer_obj = QwenLoRATrainer(model_path, dataset_path)
    trainer = trainer_obj.train()
    
    # 测试推理效果
    trainer_obj.test_inference()

if __name__ == "__main__":
    main()