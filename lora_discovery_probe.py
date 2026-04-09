import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import warnings

warnings.filterwarnings("ignore")


def run_discovery():
    print("=" * 60)
    print("🔬 启动 DINOv3 底层结构解剖探针...")
    print("=" * 60)

    dino_path = "/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/checkpoints/dinov3_vitb16"

    try:
        print(f"📦 正在裸载 DINOv3 基础模型 (跳过 PEFT)...")
        config = AutoConfig.from_pretrained(dino_path, local_files_only=True)
        model = AutoModel.from_pretrained(dino_path, config=config, local_files_only=True)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print("\n🔍 扫描第一个 Transformer Block 的 nn.Linear 命名...")
    print("-" * 60)

    # 记录找到的 linear 层名字
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.append(name)

    # 我们只需要看前 15 个，这绝对足够覆盖 layer 0 的所有门控结构了
    if not linear_names:
        print("❌ 没有找到任何 nn.Linear 层！这极其反常，请确认模型是否真的是 ViT 架构。")
        return

    for name in linear_names[:15]:
        print(f" ▶ {name}")

    print("-" * 60)
    print("🎯 请把上面打印出的带有 'layer.0' 或 'blocks.0' 的名字复制发给我。")
    print("我将根据这些真实的组件名称，为你重新定制 lora_utils.py 里的 target_modules。")
    print("=" * 60)


if __name__ == "__main__":
    run_discovery()