import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from ..resnet import ResNet, Bottleneck, BasicBlock
except ImportError:
    import sys

    sys.path.append("..")
    from mixed.modeling.resnet import ResNet, Bottleneck, BasicBlock


class PrunableConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(PrunableConv2d, self).__init__(*args, **kwargs)
        self.register_buffer("prune_mask", torch.ones_like(self.weight))
        self.prune_flag = False

    def forward(self, input):
        if not self.prune_flag:
            weight = self.weight
        else:
            weight = self.weight * self.prune_mask

        return self._conv_forward(input, weight, self.bias)

    def set_prune_flag(self, flag):
        self.prune_flag = flag


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.proj = PrunableConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, dino_feat, target_size):
        x = self.proj(dino_feat)
        x = self.bn(x)
        x = self.act(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class SideResNet(ResNet):
    def __init__(self, layers=[3, 4, 6, 3], block=Bottleneck, num_classes=5, dino_embed_dim=768, pretrained_path=None):
        super(SideResNet, self).__init__(block, layers, num_classes=num_classes)

        self._replace_conv_layers()

        if pretrained_path:
            print(f">> [SideResNet] Loading pretrained weights from: {pretrained_path}")
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

                print(f">> [SideResNet] Pretrained weights loaded successfully.")
                real_missing = [k for k in missing_keys if not k.startswith('fc') and not k.startswith('fusion') and not k.endswith('prune_mask')]
                if len(real_missing) > 0:
                    print(f"   [Warning] Missing keys: {real_missing}")

            except Exception as e:
                print(f">> [Error] Failed to load pretrained weights: {e}")
                print("   Continuing with random initialization.")
        else:
            print(">> [Warning] No pretrained_path provided! SideResNet is training from scratch.")

        self.fusion1 = FusionBlock(dino_embed_dim, 256)
        self.fusion2 = FusionBlock(dino_embed_dim, 512)
        self.fusion3 = FusionBlock(dino_embed_dim, 1024)
        self.fusion4 = FusionBlock(dino_embed_dim, 2048)

        self.out_dim = 512 * block.expansion
        self.fc = nn.Linear(self.out_dim, num_classes)

        self._init_new_modules()

    def _replace_conv_layers(self):
        def replace_layer(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d) and not isinstance(child, PrunableConv2d):
                    new_layer = PrunableConv2d(child.in_channels, child.out_channels, child.kernel_size, child.stride, child.padding, child.dilation, child.groups, child.bias is not None, child.padding_mode)
                    new_layer.weight.data = child.weight.data
                    if child.bias is not None:
                        new_layer.bias.data = child.bias.data
                    setattr(module, name, new_layer)
                else:
                    replace_layer(child)

        replace_layer(self)

    def _init_new_modules(self):
        for m in [self.fusion1, self.fusion2, self.fusion3, self.fusion4, self.fc]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def set_prune_flag(self, flag):
        for m in self.modules():
            if isinstance(m, PrunableConv2d):
                m.set_prune_flag(flag)

    def forward(self, x, dino_tokens, return_features=False):
        B, N, D = dino_tokens.shape

        H_grid = int(math.sqrt(N))
        N_spatial = H_grid * H_grid
        N_special = N - N_spatial

        assert N_special < H_grid, f"Error: Found too many special tokens ({N_special}). Total: {N}, Grid: {H_grid}x{H_grid}."

        dino_feat = dino_tokens[:, N_special:, :]
        dino_map = dino_feat.permute(0, 2, 1).reshape(B, D, H_grid, H_grid)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = x + self.fusion1(dino_map, x.shape[-2:])

        x = self.layer2(x)
        x = x + self.fusion2(dino_map, x.shape[-2:])

        x = self.layer3(x)
        x = x + self.fusion3(dino_map, x.shape[-2:])

        x = self.layer4(x)
        x = x + self.fusion4(dino_map, x.shape[-2:])

        x = self.global_avgpool(x)

        features = torch.flatten(x, 1)
        logits = self.fc(features)

        if return_features:
            return logits, features
        return logits