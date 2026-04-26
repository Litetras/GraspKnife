import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# ================== 【新增：全局单例缓存变量】 ==================
_GLOBAL_QWEN_CACHE = None
# =============================================================

class QwenTextEncoder(nn.Module):
    def __init__(self, model_id="/home/zyp/models/qwen/Qwen2___5-3B-Instruct", target_dim=512, use_4bit=True):
        super().__init__()
        
        # ================== 【新增：拦截重复加载逻辑】 ==================
        global _GLOBAL_QWEN_CACHE
        if _GLOBAL_QWEN_CACHE is not None:
            print("\n🚀 [系统提示] 检测到重复实例化，触发共享模式：直接复用已加载的 Qwen，节省显存！\n")
            self.tokenizer = _GLOBAL_QWEN_CACHE['tokenizer']
            self.qwen = _GLOBAL_QWEN_CACHE['qwen']
            self.mlp_projector = _GLOBAL_QWEN_CACHE['mlp_projector']
            return  # 直接 return，跳过下面的加载代码！
        # ==============================================================

        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True
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

        print(f"Loading Qwen model {model_id} for the FIRST time...")
        
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        self.qwen.print_trainable_parameters()
        self.qwen.gradient_checkpointing_enable()

        # 4. 定义 MLP 投影层
        qwen_hidden_size = self.qwen.config.hidden_size
        self.mlp_projector = nn.Sequential(
            nn.Linear(qwen_hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, target_dim)
        ).to(torch.bfloat16).to(self.qwen.device)

        # ================== 【新增：首次加载完成后，写入全局缓存】 ==================
        _GLOBAL_QWEN_CACHE = {
            'tokenizer': self.tokenizer,
            'qwen': self.qwen,
            'mlp_projector': self.mlp_projector
        }
        # =========================================================================
        
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