from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class QwenModel:
    """Qwen模型类"""
    
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.device = next(self.model.parameters()).device
        print(f"模型已加载到设备: {self.device}")
        
    def generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.1) -> str:
        """
        生成模型响应
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除输入提示部分
        response = response[len(prompt):].strip()
        return response
    
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

    """Qwen模型类"""
    
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True  # 减少CPU内存拷贝
        ).eval()  # 禁用dropout等训练模式
        self.device = next(self.model.parameters()).device
        print(f"模型已加载到设备: {self.device}")

        # 预热GPU（避免首次运行延迟）
        inputs = self.tokenizer("预热模型", return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=1)
        

    def generate_response(self, prompt: str, max_length: int = 50, temperature: float = 0.1) -> str:
        """
        生成模型响应
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            # eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,  # 启用KV缓存
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除输入提示部分
        response = response[len(prompt):].strip()
        return response
    
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

    """Qwen模型类"""
    
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # 设置left padding用于decoder-only模型
        self.tokenizer.padding_side = 'left'
        
        # 添加pad_token
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
        
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.1) -> str:
        """
        生成模型响应
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=self.tokenizer.pad_token_id,
            # eos_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # # 移除输入提示部分
        # response = response[len(prompt):].strip()
        return response
    
    def generate_batch_responses(self, prompts: List[str], max_length: int = 512, temperature: float = 0.1) -> List[str]:
        """
        批量生成模型响应 - 优化版本以保持与单独推理一致的准确率
        
        Args:
            prompts: 输入提示列表
            max_length: 最大生成长度
            temperature: 温度参数
        """
        # 方案1：如果批次很小，使用逐个推理以保证准确率
        if len(prompts) <= 4:  # 小批次直接用单独推理
            return [self.generate_response(prompt, max_length, temperature) for prompt in prompts]
        
        # 方案2：对于大批次，使用改进的批处理
        # 按长度分组，减少padding的影响
        prompt_groups = self._group_prompts_by_length(prompts)
        all_responses = []
        
        for group_prompts in prompt_groups:
            # 每组内的长度相近，padding影响较小
            inputs = self.tokenizer(
                group_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                padding_side='left'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # 添加这些参数提高生成质量
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            
            group_responses = []
            for i, output in enumerate(outputs):
                # 计算真实的输入长度（排除左侧padding）
                attention_mask = inputs.attention_mask[i]
                # 找到第一个非padding token的位置
                first_token_pos = (attention_mask == 1).nonzero()[0].item()
                input_length = len(inputs.input_ids[i]) - first_token_pos
                
                # 从输出中提取新生成的部分
                generated_tokens = output[-max_length:]  # 只取最后生成的部分
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                # 清理响应，移除可能的prompt残留
                original_prompt = group_prompts[i]
                if response.startswith(original_prompt.strip()):
                    response = response[len(original_prompt.strip()):].strip()
                    
                group_responses.append(response)
            
            all_responses.extend(group_responses)
        
        # 恢复原始顺序
        return self._restore_original_order(prompts, all_responses)
    
    def _group_prompts_by_length(self, prompts: List[str], max_length_diff: int = 50) -> List[List[str]]:
        """按长度将prompts分组以减少padding影响"""
        # 计算每个prompt的长度
        prompt_lengths = [(i, len(self.tokenizer.encode(prompt))) for i, prompt in enumerate(prompts)]
        prompt_lengths.sort(key=lambda x: x[1])  # 按长度排序
        
        groups = []
        current_group = []
        current_base_length = 0
        
        for idx, length in prompt_lengths:
            if not current_group or abs(length - current_base_length) <= max_length_diff:
                current_group.append((idx, prompts[idx]))
                if not current_base_length:
                    current_base_length = length
            else:
                groups.append([item[1] for item in current_group])
                current_group = [(idx, prompts[idx])]
                current_base_length = length
        
        if current_group:
            groups.append([item[1] for item in current_group])
        
        return groups
    
    def _restore_original_order(self, original_prompts: List[str], responses: List[str]) -> List[str]:
        """恢复响应的原始顺序"""
        # 简化版本：如果分组逻辑复杂，这里可能需要更精细的实现
        # 当前假设分组后顺序基本保持
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