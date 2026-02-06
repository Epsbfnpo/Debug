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


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, dino_feat, target_size):
        x = self.proj(dino_feat)
        x = self.bn(x)
        x = self.act(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class SideResNet(ResNet):
    def __init__(self, layers=[3, 4, 6, 3], block=Bottleneck, num_classes=5, dino_embed_dim=768):
        super(SideResNet, self).__init__(block, layers, num_classes=num_classes)

        self.fusion1 = FusionBlock(dino_embed_dim, 256)
        self.fusion2 = FusionBlock(dino_embed_dim, 512)
        self.fusion3 = FusionBlock(dino_embed_dim, 1024)
        self.fusion4 = FusionBlock(dino_embed_dim, 2048)

        self.out_dim = 512 * block.expansion
        self.fc = nn.Linear(self.out_dim, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, dino_tokens, return_features=False):
        B, N, D = dino_tokens.shape

        H_grid = int(math.sqrt(N))

        N_spatial = H_grid * H_grid

        N_special = N - N_spatial

        assert N_special < H_grid, f"Error: Found too many special tokens ({N_special}). Total: {N}, Grid: {H_grid}x{H_grid}. Check input resolution."

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