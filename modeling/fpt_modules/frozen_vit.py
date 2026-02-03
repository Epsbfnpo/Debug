import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from peft import get_peft_model, LoraConfig


class FrozenViT(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.last_attention_map = None

        for param in self.model.parameters():
            param.requires_grad = False

    def _patch_last_layer_attention(self):
        layers = None
        if hasattr(self.model, "layer"):
            layers = self.model.layer
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model.model, "layer"):
            layers = self.model.base_model.model.layer
        elif hasattr(self.model, "layers"):
            layers = self.model.layers

        if layers is None:
            raise AttributeError("CRITICAL: Could not find 'layer' module in the model!")

        last_layer = layers[-1]

        if not hasattr(last_layer, 'attention'):
            raise AttributeError("CRITICAL: Last layer does not have 'attention' module!")

        target_attention_module = last_layer.attention

        original_forward = target_attention_module.forward

        def patched_forward(*args, **kwargs):
            kwargs['output_attentions'] = True

            outputs = original_forward(*args, **kwargs)

            if isinstance(outputs, tuple) and len(outputs) > 1:
                self.last_attention_map = outputs[1]
            else:
                pass

            return outputs

        print(f">> [System] Monkey-patching the last layer's Attention module to force capture map...")
        target_attention_module.forward = patched_forward

    def forward(self, x, interpolate_pos_encoding=False, **kwargs):
        embeddings_module = None
        if hasattr(self.model, "embeddings"):
            embeddings_module = self.model.embeddings
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model.model, "embeddings"):
            embeddings_module = self.model.base_model.model.embeddings

        if embeddings_module is None:
            raise AttributeError("Could not find 'embeddings' module.")

        embed_kwargs = {}
        if hasattr(embeddings_module, "forward"):
            import inspect
            sig = inspect.signature(embeddings_module.forward)
            if "interpolate_pos_encoding" in sig.parameters:
                embed_kwargs["interpolate_pos_encoding"] = interpolate_pos_encoding
        hidden_states = embeddings_module(x, **embed_kwargs)

        position_embeddings = None
        rope_module = None
        if hasattr(self.model, "rope_embeddings"):
            rope_module = self.model.rope_embeddings
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model.model, "rope_embeddings"):
            rope_module = self.model.base_model.model.rope_embeddings

        if rope_module is not None:
            position_embeddings = rope_module(x)

        layers = None
        if hasattr(self.model, "layer"):
            layers = self.model.layer
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model.model, "layer"):
            layers = self.model.base_model.model.layer
        elif hasattr(self.model, "layers"):
            layers = self.model.layers

        all_hidden_states = [hidden_states]

        self.last_attention_map = None

        for i, layer_module in enumerate(layers):
            layer_kwargs = {}
            if position_embeddings is not None:
                try:
                    import inspect
                    if "position_embeddings" in inspect.signature(layer_module.forward).parameters:
                        layer_kwargs["position_embeddings"] = position_embeddings
                except:
                    pass

            hidden_states = layer_module(hidden_states, **layer_kwargs)

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            all_hidden_states.append(hidden_states)

        norm_module = None
        if hasattr(self.model, "norm"):
            norm_module = self.model.norm
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model.model, "norm"):
            norm_module = self.model.base_model.model.norm

        if norm_module is not None:
            hidden_states = norm_module(hidden_states)

        features = torch.stack(all_hidden_states)

        if self.last_attention_map is None:
            print(">> [WARNING] Attention Map not captured! Is the patch working?")

        attentions = [None] * (len(layers) - 1) + [self.last_attention_map]

        return None, features, tuple(attentions)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)

        model = cls(config)

        pretrained_backbone = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        model.model = pretrained_backbone

        # print(f"Injecting LoRA adapters into FrozenViT (Backbone)...")
        # peft_config = LoraConfig(...)
        # model.model = get_peft_model(model.model, peft_config)

        print(f">> [System] Loaded pure FrozenViT backbone (No LoRA injected).")

        model._patch_last_layer_attention()

        return model