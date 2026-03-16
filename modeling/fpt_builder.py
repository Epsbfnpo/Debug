import torch
from transformers import ViTConfig

from .fpt_core.bridge import FineGrainedPromptTuning, FusionModule
from .fpt_core.side_vit import ViTForImageClassification as SideViT
from .fpt_core.frozen_vit import ViTForImageClassification as FrozenViT


def build_fpt_models(
    pretrained_path="google/vit-base-patch16-224",
    num_classes=5,
    layers_to_extract=None,
    token_ratio=0.1,
    token_imp="global",
    side_reduction_ratio=4,
    prompt_reduction_ratio=4,
    num_prompts=4,
    side_input_size=224,
):
    if layers_to_extract is None:
        layers_to_extract = [3, 7, 11]

    frozen_config = ViTConfig.from_pretrained(pretrained_path)
    frozen_config.token_imp = token_imp
    frozen_config.token_ratio = token_ratio
    frozen_config.layers_to_extract = layers_to_extract
    frozen_config.num_labels = num_classes

    frozen_encoder = FrozenViT.from_pretrained(pretrained_path, config=frozen_config)
    frozen_encoder.eval()
    for p in frozen_encoder.parameters():
        p.requires_grad = False

    num_layers = len(layers_to_extract)
    side_dimension = frozen_config.hidden_size // side_reduction_ratio
    prompts_dim = frozen_config.hidden_size // prompt_reduction_ratio

    fusion_module = FusionModule(
        num_layers=num_layers,
        in_dim=frozen_config.hidden_size,
        out_dim=side_dimension,
        num_heads=frozen_config.num_attention_heads,
        num_prompts=num_prompts,
        prompt_dim=prompts_dim,
        prompt_norm=True,
        prompt_proj=False,
    )

    side_config = ViTConfig.from_pretrained(
        pretrained_path,
        num_hidden_layers=num_layers,
        hidden_size=side_dimension,
        intermediate_size=side_dimension * 4,
        image_size=side_input_size,
        num_labels=num_classes,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    side_encoder = SideViT(side_config)

    trainable_model = FineGrainedPromptTuning(side_encoder, fusion_module)
    return frozen_encoder, trainable_model
