import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

def inject_dinov3_lora(model, rank=8, alpha=16.0, dropout=0.0):

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
    ]

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    peft_model = get_peft_model(model, config)

    print(f"✅ [LoRA Injected] 已全面覆盖核心层: {target_modules} | Rank: {rank}")
    peft_model.print_trainable_parameters()

    return peft_model
