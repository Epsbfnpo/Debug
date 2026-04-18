import torch
import torch.nn as nn
import re

def inject_dinov3_lora(model, rank=8, alpha=8.0, dropout=0.1):
    """
    精确打击 DINOv3 (HuggingFace 格式) 的核心门控：
    Attention 层: q_proj, k_proj, v_proj, o_proj
    MLP 层: up_proj, down_proj
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("🔥 严重错误: 请先安装 peft 库 (pip install peft)。这是微调大模型的标准武器。")

    # 统一命中注意力/MLP层名，随后再二次冻结为仅最后4个block可训练
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=[
            "qkv",
            "fc1",
            "fc2",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    # 注入 LoRA
    peft_model = get_peft_model(model, config)

    # 仅开放最后4个Transformer Block（8~11）的LoRA参数
    deep_block_pattern = re.compile(r".*(blocks|layer)\.(8|9|10|11)\..*")
    trainable_params = 0
    total_params = 0
    for name, param in peft_model.named_parameters():
        total_params += param.numel()
        is_lora = "lora" in name.lower()
        is_deep_lora = is_lora and deep_block_pattern.match(name) is not None
        is_classifier = "classifier" in name.lower()
        if is_deep_lora or is_classifier:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f"✅ [LoRA Injected] 仅训练最后4层 Block(8-11) | Rank: {rank} | Alpha: {alpha}")
    print(f"📊 [LoRA Params] 训练参数: {trainable_params / 1e6:.2f}M / 总参数: {total_params / 1e6:.2f}M")
    peft_model.print_trainable_parameters()

    return peft_model
