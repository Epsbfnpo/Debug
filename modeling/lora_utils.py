import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def inject_dinov3_lora(model, rank=8, alpha=16.0, dropout=0.0):
    """
    全层精准注入 LoRA (HuggingFace Transformers 适配版)
    根据 info.txt 的探针结果，精准锁定 HuggingFace DINOv3 的：
    - Attention 层: q_proj, k_proj, v_proj, o_proj
    - MLP 层: up_proj, down_proj
    解锁 DINOv3 映射医学底层纹理的能力。
    """
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
