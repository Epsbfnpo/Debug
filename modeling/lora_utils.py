import torch
import torch.nn as nn

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

    # 🔪 致命修正：完全匹配探针扫出来的底层组件名称
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

    # 强制确保只有 LoRA 参数可训练
    trainable_params = 0
    total_params = 0
    for name, param in peft_model.named_parameters():
        total_params += param.numel()
        if "lora" in name.lower():
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f"✅ [LoRA Injected] 目标层: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj | Rank: {rank} | Alpha: {alpha}")
    print(f"📊 [LoRA Params] 训练参数: {trainable_params / 1e6:.2f}M / 总参数: {total_params / 1e6:.2f}M")

    return peft_model
