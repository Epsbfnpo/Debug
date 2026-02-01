from .resnet import resnet18, resnet50, resnet101
from .nets import *
from .nets import FPTPlusNet
import torch


def get_net(cfg):
    # 针对 ERM 和 GDRNet 算法，我们允许替换 Backbone
    if cfg.ALGORITHM == 'ERM' or cfg.ALGORITHM == 'GDRNet':
        # [新增] 如果配置文件指定 BACKBONE 为 'FPT'，则使用 FPTPlusNet
        if cfg.BACKBONE == 'FPT':
            net = FPTPlusNet(cfg)
        else:
            # 否则使用传统的 ResNet (resnet18/50/101)
            net = get_backbone(cfg)

    # [新增] 如果你定义了一个名为 'FPT_DG' 的专用算法名称，也走这里
    elif cfg.ALGORITHM == 'FPT_DG':
        net = FPTPlusNet(cfg)

    # 以下保持原有逻辑不变
    elif cfg.ALGORITHM == 'GREEN':
        net = SoftLabelGCN(cfg)
    elif cfg.ALGORITHM == 'CABNet':
        net = CABNet(cfg)
    elif cfg.ALGORITHM == 'MixupNet':
        net = MixupNet(cfg)
    elif cfg.ALGORITHM == 'MixStyleNet':
        net = MixStyleNet(cfg)
    elif cfg.ALGORITHM == 'Fishr' or cfg.ALGORITHM == 'DRGen':
        net = FishrNet(cfg)
    else:
        raise ValueError(f'Wrong algorithm type: {cfg.ALGORITHM}')
    return net


def get_backbone(cfg):
    if cfg.BACKBONE == 'resnet18':
        model = resnet18(pretrained=True)
    elif cfg.BACKBONE == 'resnet50':
        model = resnet50(pretrained=True)
    elif cfg.BACKBONE == 'resnet101':
        model = resnet101(pretrained=True)
    else:
        raise ValueError(f'Wrong backbone type: {cfg.BACKBONE}')
    return model


def get_classifier(out_feature_size, cfg):
    return torch.nn.Linear(out_feature_size, cfg.DATASET.NUM_CLASSES)