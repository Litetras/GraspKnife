import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import os


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


class QwenTextEncoder(nn.Module):
    def __init__(self, model_id="/home/zyp/models/qwen/Qwen2___5-3B-Instruct", target_dim=512, use_4bit=True):
        super().__init__()
        self.max_length = int(os.environ.get("GRASPGEN_QWEN_MAX_LENGTH", "64"))
        self.cache_tokenizer = _env_flag("GRASPGEN_QWEN_CACHE_TOKENIZER", "1")
        self.dedup_batch_text = _env_flag("GRASPGEN_QWEN_DEDUP_BATCH", "1")
        self.gradient_checkpointing = _env_flag(
            "GRASPGEN_QWEN_GRADIENT_CHECKPOINTING",
            "0",
        )
        self.use_backbone_only = _env_flag("GRASPGEN_QWEN_USE_BACKBONE_ONLY", "1")
        self._token_cache = {}
        
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
        self.qwen.config.use_cache = False

        if self.gradient_checkpointing:
            # 开启梯度检查点：更省显存，但会显著变慢。默认关闭，需要时用环境变量打开。
            self.qwen.gradient_checkpointing_enable()
            print("[QwenTextEncoder] gradient checkpointing: ON")
        else:
            if hasattr(self.qwen, "gradient_checkpointing_disable"):
                self.qwen.gradient_checkpointing_disable()
            print("[QwenTextEncoder] gradient checkpointing: OFF")

        self.qwen_backbone = self._resolve_qwen_backbone()
        if self.use_backbone_only and self.qwen_backbone is not None:
            print("[QwenTextEncoder] using Qwen backbone hidden states (no lm_head logits).")
        elif self.use_backbone_only:
            print("[QwenTextEncoder] backbone path unavailable, falling back to full Qwen forward.")

        print(
            "[QwenTextEncoder] speed config: "
            f"token_cache={self.cache_tokenizer}, dedup_batch={self.dedup_batch_text}, "
            f"max_length={self.max_length}, backbone_only={self.use_backbone_only}"
        )

        # 4. 定义 MLP 投影层 (Qwen2.5-3B 的 hidden_size 是 2048)
        qwen_hidden_size = self.qwen.config.hidden_size
        self.mlp_projector = nn.Sequential(
            nn.Linear(qwen_hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, target_dim)
        ).to(torch.bfloat16).to(self.qwen.device)

    def _resolve_qwen_backbone(self):
        """Return the transformer backbone so feature extraction avoids lm_head logits."""
        candidates = []
        if hasattr(self.qwen, "base_model"):
            candidates.append(getattr(self.qwen, "base_model"))
        candidates.append(self.qwen)

        for candidate in candidates:
            obj = candidate
            for attr_path in [
                ("model", "model"),
                ("model",),
            ]:
                obj = candidate
                ok = True
                for attr in attr_path:
                    if not hasattr(obj, attr):
                        ok = False
                        break
                    obj = getattr(obj, attr)
                if ok and obj is not self.qwen and callable(obj):
                    return obj
        return None

    def _get_model_device(self):
        return getattr(self.qwen, "device", next(self.qwen.parameters()).device)

    def _tokenize_uncached_single(self, prompt):
        encoded = self.tokenizer(
            str(prompt),
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0).cpu(),
            "attention_mask": encoded["attention_mask"].squeeze(0).cpu(),
        }

    def _tokenize_text_list(self, text_list):
        if not self.cache_tokenizer:
            return self.tokenizer(
                text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

        encoded_items = []
        for prompt in text_list:
            cache_key = str(prompt)
            encoded = self._token_cache.get(cache_key)
            if encoded is None:
                encoded = self._tokenize_uncached_single(cache_key)
                self._token_cache[cache_key] = encoded
            encoded_items.append(encoded)

        max_len = max(item["input_ids"].numel() for item in encoded_items)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full(
            (len(encoded_items), max_len),
            fill_value=pad_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(encoded_items), max_len),
            dtype=torch.long,
        )
        for row, item in enumerate(encoded_items):
            length = item["input_ids"].numel()
            input_ids[row, :length] = item["input_ids"]
            attention_mask[row, :length] = item["attention_mask"]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _forward_hidden_states(self, inputs):
        if self.use_backbone_only and self.qwen_backbone is not None:
            outputs = self.qwen_backbone(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state

        outputs = self.qwen(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def _encode_no_dedup(self, text_list):
        device = self._get_model_device()
        inputs = self._tokenize_text_list(text_list)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        last_hidden_state = self._forward_hidden_states(inputs)

        batch_size = last_hidden_state.shape[0]
        sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
        pooled_features = last_hidden_state[batch_indices, sequence_lengths]

        target_dtype = self.mlp_projector[0].weight.dtype
        pooled_features = pooled_features.to(target_dtype)
        projected_features = self.mlp_projector(pooled_features)
        return projected_features.to(torch.float32)

    def forward(self, text_list):
        """
        输入: text_list (例如 ["up handle", "down blade"])
        输出: shape 为 [batch_size, 512] 的特征向量
        """
        text_list = [str(text) for text in text_list]
        if not self.dedup_batch_text:
            return self._encode_no_dedup(text_list)

        unique_texts = []
        inverse_indices = []
        text_to_unique_idx = {}
        for text in text_list:
            idx = text_to_unique_idx.get(text)
            if idx is None:
                idx = len(unique_texts)
                text_to_unique_idx[text] = idx
                unique_texts.append(text)
            inverse_indices.append(idx)

        unique_features = self._encode_no_dedup(unique_texts)
        inverse_indices = torch.tensor(
            inverse_indices,
            dtype=torch.long,
            device=unique_features.device,
        )
        return unique_features[inverse_indices]
