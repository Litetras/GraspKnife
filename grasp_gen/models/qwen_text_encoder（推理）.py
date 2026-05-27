import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 全局缓存只保存重型组件（Qwen主干），不缓存 mlp_projector
_GLOBAL_QWEN_CACHE = None

class QwenTextEncoder(nn.Module):
    def __init__(self, model_id="/home/zyp/models/qwen/Qwen2___5-3B-Instruct", target_dim=512, use_4bit=True):
        super().__init__()
        
        global _GLOBAL_QWEN_CACHE

        if _GLOBAL_QWEN_CACHE is not None:
            print("\n🚀 [系统提示] 复用已加载的 Qwen 主干，节省显存！\n")
            self.tokenizer = _GLOBAL_QWEN_CACHE['tokenizer']
            self.qwen      = _GLOBAL_QWEN_CACHE['qwen']
            # ✅ 关键修复：mlp_projector 不从缓存读取，每个实例独立创建
            qwen_hidden_size = self.qwen.config.hidden_size
            self.mlp_projector = nn.Sequential(
                nn.Linear(qwen_hidden_size, 1024),
                nn.GELU(),
                nn.Linear(1024, target_dim)
            ).to(torch.bfloat16).to(self.qwen.device)
            return  # 只跳过 Qwen 加载，不跳过 mlp_projector 创建

        # ---- 首次加载 ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

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

        # mlp_projector 在首次加载时也正常创建
        qwen_hidden_size = self.qwen.config.hidden_size
        self.mlp_projector = nn.Sequential(
            nn.Linear(qwen_hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, target_dim)
        ).to(torch.bfloat16).to(self.qwen.device)

        # ✅ 全局缓存只保存 Qwen 主干，不包含 mlp_projector
        _GLOBAL_QWEN_CACHE = {
            'tokenizer': self.tokenizer,
            'qwen':      self.qwen,
        }

    def forward(self, text_list):
        inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self.qwen.device)

        outputs = self.qwen(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]

        batch_size = last_hidden_state.shape[0]
        sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
        pooled_features = last_hidden_state[torch.arange(batch_size), sequence_lengths]

        target_dtype = self.mlp_projector[0].weight.dtype
        pooled_features = pooled_features.to(target_dtype)
        projected_features = self.mlp_projector(pooled_features)

        return projected_features.to(torch.float32)