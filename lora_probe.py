import torch
import warnings

# 忽略一些无关紧要的 HuggingFace 警告
warnings.filterwarnings("ignore")

from modeling.nets import DINOv3Wrapper


def run_lora_probe():
    print("=" * 50)
    print("🔍 启动 DINOv3 + LoRA 深度探针...")
    print("=" * 50)

    # ⚠️ 请确保这里的路径与你 configs/代码 中的实际路径一致
    # 你的代码里写的是 "/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/checkpoints/dinov3_vitb16"
    dino_path = "/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/checkpoints/dinov3_vitb16"

    try:
        print(f"📦 正在加载 DINOv3 基础模型并注入 LoRA (Rank=8)...")
        # 强制使用与真实环境一致的配置进行初始化
        wrapper = DINOv3Wrapper(
            local_path=dino_path,
            lora_r=8,
            lora_alpha=16.0,
            lora_dropout=0.0,
            num_drts=4,
            use_grad_checkpointing=False
        )
    except Exception as e:
        print(f"\n❌ [致命错误] 模型加载或注入失败！请检查模型路径或是否正确安装了 peft 库。")
        print(f"错误详情: {e}")
        return

    # 1. 参数级审计 (Parameter Audit)
    print("\n📊 [第一关：参数梯度审计]")
    trainable_params = 0
    all_param = 0
    trainable_names = []
    frozen_names = []

    for name, param in wrapper.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            trainable_names.append(name)
        else:
            frozen_names.append(name)

    print(f"  ▶ 总参数量:      {all_param / 1e6:.2f} M")
    print(f"  ▶ 可训练参数量:  {trainable_params / 1e6:.2f} M")
    print(f"  ▶ 可训练比例:    {100 * trainable_params / all_param:.4f} %")

    # 检查是否有非法参数被解冻
    illegal_unfrozen = [name for name in trainable_names if 'lora' not in name.lower() and 'drt' not in name.lower()]
    if illegal_unfrozen:
        print("\n🚨 [警报] 发现非法解冻的参数 (非 LoRA 且 非 DRT)！你的基础大模型可能被破坏了！")
        for name in illegal_unfrozen[:5]:
            print(f"  - {name}")
    else:
        print("  ✅ 基础参数安全冻结，只有 LoRA 和 DRTs 处于激活状态。")

    # 2. 挂载点审计 (Mount Point Audit)
    print("\n🎯 [第二关：LoRA 精确打击面审计]")
    lora_layers = set()
    for name in trainable_names:
        if 'lora' in name.lower():
            # 提取出层级名字，例如 'model.encoder.layer.0.attention.attention.query'
            base_layer = name.split('.lora_')[0]
            lora_layers.add(base_layer)

    # 按照论文设定，必须包含这四种类型的挂载点
    expected_targets = ['qkv', 'proj', 'fc1', 'fc2']
    found_targets = {t: 0 for t in expected_targets}

    for layer in lora_layers:
        for target in expected_targets:
            if target in layer:
                found_targets[target] += 1

    for target, count in found_targets.items():
        if count > 0:
            print(f"  ✅ 命中目标层 [{target}]: 共检测到 {count} 处注入点 (如参数对齐，应为 12x2 或 24x2)")
        else:
            print(f"  ❌ 丢失目标层 [{target}]: 你的 LoRA 根本没有挂载到该门控上！")

    if all(count > 0 for count in found_targets.values()):
        print("  🏆 完美！论文指定的四个命门 (qkv, proj, fc1, fc2) 已被全部精准锁定。")

    # 3. 计算图压力测试 (Forward Pressure Test)
    print("\n⚡ [第三关：计算图连通性测试]")
    # 模拟真实输入，bs=2, 3通道, 224x224 分辨率
    dummy_input = torch.randn(2, 3, 224, 224)

    try:
        if torch.cuda.is_available():
            wrapper = wrapper.cuda()
            dummy_input = dummy_input.cuda()

        outputs = wrapper(dummy_input)
        feature_shape = outputs.last_hidden_state.shape
        print(f"  ✅ 前向传播测试通过！计算图未崩溃。")
        print(f"  ▶ 提取到的隐层特征维度: {feature_shape}")

        # 1(CLS) + 4(DRTs) + 196(14x14 Patch) = 201
        # (如果是 ViT-B/16 在 224x224 分辨率下)
        expected_seq_len = 1 + 4 + (224 // 16) * (224 // 16)
        if feature_shape[1] == expected_seq_len:
            print(f"  ✅ Token 序列长度 ({feature_shape[1]}) 与数学预期完全一致！")
        else:
            print(f"  ⚠️ Token 序列长度为 {feature_shape[1]}，请确认这是你预期的 Patch 数量。")

    except Exception as e:
        print(f"\n❌ [致命错误] 前向传播时发生计算图崩溃！")
        print(f"错误详情: {e}")
        return

    print("\n" + "=" * 50)
    print("🚀 探针诊断完毕。如果上述全绿，你可以放心地把主程序提交到计算节点了。")
    print("=" * 50)


if __name__ == "__main__":
    run_lora_probe()