import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        self.in_features = in_f
        self.out_features = out_f
        self.bias = base.bias
        self.r = r
        self.alpha = alpha
        self.scale = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.lora_A = nn.Linear(in_f, r, bias=False)
            self.lora_B = nn.Linear(r, out_f, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            if self.lora_A.weight.device != x.device:
                self.lora_A.to(x.device)
                self.lora_B.to(x.device)
            out = out + self.scale * self.lora_B(self.dropout(self.lora_A(x)))
        return out

def inject_lora_dinov3(model: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
    target_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            attr_name = name.split('.')[-1]
            if attr_name in target_keywords:
                parent_name = ".".join(name.split('.')[:-1])
                if parent_name == "":
                    continue
                parent = model
                for p in parent_name.split('.'):
                    parent = getattr(parent, p)
                print(f"Injecting LoRA into: {name}")
                setattr(parent, attr_name, LoRALinear(module, r, alpha, dropout))
    return model