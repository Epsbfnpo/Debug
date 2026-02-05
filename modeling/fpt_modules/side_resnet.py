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
        self.out_dim = 2048 * block.expansion

    def forward(self, x, dino_tokens):
        B, N, D = dino_tokens.shape
        if int(math.sqrt(N)) ** 2 == N:
            H_grid = int(math.sqrt(N))
            dino_feat = dino_tokens
        else:
            N_patches = N - 1
            H_grid = int(math.sqrt(N_patches))
            dino_feat = dino_tokens[:, 1:, :]  # (B, N-1, D)

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

        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # (B, 2048)

        logits = self.fc(features)  # (B, num_classes)

        return logits, features


def side_resnet50(num_classes=5, dino_embed_dim=768, **kwargs):
    model = SideResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, dino_embed_dim=dino_embed_dim, **kwargs)
    return model