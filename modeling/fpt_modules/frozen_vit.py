import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
import math

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.out_features = original_layer.out_features
        self.in_features = original_layer.in_features

        # 1. 冻结原始参数
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # 2. 定义 LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros((rank, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank)))
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out_orig = self.original_layer(x)
        out_lora = (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return out_orig + out_lora

class FrozenViT(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = None

    @classmethod
    def from_pretrained(cls, model_path, use_lora=True, lora_rank=8, **kwargs):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        if 'output_hidden_states' in kwargs:
            config.output_hidden_states = kwargs.pop('output_hidden_states')
        if 'output_attentions' in kwargs:
            config.output_attentions = kwargs.pop('output_attentions')


        model = cls(config)

        print(f"Loading ViT Backbone from: {model_path}")
        model.model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True, **kwargs)

        if use_lora:
            print(f">> [Optimization] Enabling Gradient Checkpointing for LoRA training...")
            model.model.gradient_checkpointing_enable()

        if use_lora:
            print(f">> [DINOv3 Adapt] Injecting LoRA (Rank={lora_rank})...")
            model.inject_lora(lora_rank)

            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.model.parameters())
            print(
                f">> [DINOv3 Adapt] LoRA Enabled. Trainable Params: {trainable_params} / {all_params} ({trainable_params / all_params:.2%})")
        else:
            print(">> [DINOv3 Adapt] LoRA Disabled. Freezing entire backbone.")
            for param in model.model.parameters():
                param.requires_grad = False

        return model

    def inject_lora(self, rank):
        for param in self.model.parameters():
            param.requires_grad = False

        layers = None
        if hasattr(self.model, 'layer'):
            layers = self.model.layer
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer

        if layers is None:
            print("Warning: Could not find 'layer' module. LoRA injection failed.")
            return

        for i, block in enumerate(layers):
            if not hasattr(block, 'attention'):
                continue
            attn = block.attention

            if hasattr(attn, 'q_proj'):
                attn.q_proj = LoRALinear(attn.q_proj, rank=rank)

            if hasattr(attn, 'v_proj'):
                attn.v_proj = LoRALinear(attn.v_proj, rank=rank)

    def forward(self, x, **kwargs):
        if hasattr(self.model, 'dtype') and x.dtype != self.model.dtype:
            x = x.to(self.model.dtype)

        outputs = self.model(x, **kwargs)
        return outputs