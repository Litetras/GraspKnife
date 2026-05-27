import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class QwenTextEncoder(nn.Module):
    def __init__(self, model_id="/home/zyp/models/qwen/Qwen2___5-3B-Instruct", target_dim=512, use_4bit=True):
        super().__init__()
        
        # 1. 加载 Tokenizer（指定本地路径 + 强制断网，已清理重复代码）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True  # 【关键】强制只加载本地文件，不尝试联网
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 2. 显存优化：使用 4bit 加载 Qwen
        model_kwargs = {}
        if use_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        print(f"Loading Qwen model {model_id}...")
        
        # 加载模型（指定本地路径 + 强制断网）
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            local_files_only=True,
            **model_kwargs
        )
        
        # 3. 冻结 Qwen 原始参数并应用 LoRA
        for param in self.qwen.parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 微调注意力层
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        self.qwen.print_trainable_parameters() # 打印可训练的参数量
        
        # ================== 【新增：显存救命神器】 ==================
        # 开启梯度检查点：用计算时间换取显存空间，能省下将近 30%~50% 的显存峰值消耗！
        self.qwen.gradient_checkpointing_enable()
        # =========================================================

        # 4. 定义 MLP 投影层 (Qwen2.5-3B 的 hidden_size 是 2048)
        qwen_hidden_size = self.qwen.config.hidden_size
        self.mlp_projector = nn.Sequential(
            nn.Linear(qwen_hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, target_dim)
        ).to(torch.bfloat16).to(self.qwen.device)

    def forward(self, text_list):
        """
        输入: text_list (例如 ["up handle", "down blade"])
        输出: shape 为 [batch_size, 512] 的特征向量
        """
        # Tokenize 文本
        inputs = self.tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=64
        ).to(self.qwen.device)

        # 提取特征 (output_hidden_states=True 提取隐层状态)
        outputs = self.qwen(**inputs, output_hidden_states=True)
        
        # 取最后一层的隐藏状态
        last_hidden_state = outputs.hidden_states[-1] # Shape: [Batch, Seq_Len, Hidden_Size]

        # 提取每个句子的最后一个有效 Token（非 Padding Token）的隐藏状态
        batch_size = last_hidden_state.shape[0]
        sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
        
        # 获取汇聚后的特征: Shape: [Batch, 2048]
        pooled_features = last_hidden_state[torch.arange(batch_size), sequence_lengths]

        # ==================== 【精度冲突补丁】 ====================
        # 1. 强制对齐精度 (防止 mat1 vs mat2 报错)
        target_dtype = self.mlp_projector[0].weight.dtype
        pooled_features = pooled_features.to(target_dtype)

        # 2. 通过 MLP 降维到 512 维
        projected_features = self.mlp_projector(pooled_features)

        # 3. 强制把最终输出转回标准的 Float32
        return projected_features.to(torch.float32)
        # =========================================================