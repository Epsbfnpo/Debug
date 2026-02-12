from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch
import numpy as np
from transformers import AutoImageProcessor
import os


def get_dataset(args, cfg):
    # 1. 获取增强变换
    if cfg.ALGORITHM != 'GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
        tra_mask = None  # 非 GDRNet 模式默认没有 mask transform
    else:
        # GDRNet 模式：获取特定的预处理和增强
        train_ts, test_ts, tra_fundus, tra_mask = get_pre_FundusAug(cfg)

    batch_size = cfg.BATCH_SIZE
    drop_last = getattr(cfg, 'DROP_LAST', True)
    num_worker = getattr(cfg, 'WORKERS', 4)

    # 2. 初始化数据集

    # --- Train Set ---
    train_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='train',
        trans_basic=train_ts,  # Weak View & Mixed View Base
        trans_fundus=tra_fundus,  # Strong View (包含 FundusAug)
        trans_mask=tra_mask  # [关键] 传入 Mask 变换
    )

    train_sampler = None
    train_shuffle = True
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_worker,
        drop_last=drop_last,
        pin_memory=True,
        sampler=train_sampler
    )

    # --- Val Set ---
    val_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='val',
        trans_basic=test_ts,
        trans_mask=tra_mask  # [修复] 必须传入 trans_mask，否则返回 PIL 报错
    )

    val_sampler = None
    if args.local_rank != -1:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        sampler=val_sampler
    )

    # --- Test Set ---
    test_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='test',
        trans_basic=test_ts,
        trans_mask=tra_mask  # [修复] 必须传入 trans_mask
    )

    test_sampler = None
    if args.local_rank != -1:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        sampler=test_sampler
    )

    dataset_size = [len(train_dataset), len(val_dataset), len(test_dataset)]

    return train_loader, val_loader, test_loader, dataset_size, train_sampler


def get_transform(cfg):
    re_size = getattr(cfg.INPUT, 'SIZE', 224)
    normalize = get_normalize(cfg)

    tra_train = transforms.Compose([
        transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        normalize
    ])

    tra_test = transforms.Compose([
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor(),
        normalize
    ])

    tra_mask = transforms.Compose([
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])

    return tra_train, tra_test, tra_mask


def get_pre_FundusAug(cfg):
    aug_prob = getattr(cfg.TRANSFORM, 'AUGPROB', 0.5)
    re_size = getattr(cfg.INPUT, 'SIZE', 224)  # 512
    size = int(re_size * (256 / 224))  # 585

    normalize = get_normalize(cfg)

    # 1. Weak Augmentation
    tra_train_weak = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(re_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # 2. Strong Augmentation
    tra_train_strong = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(re_size),  # 裁剪后图片变为 512x512
        transforms.RandomHorizontalFlip(),

        FundusAug.Compose([
            FundusAug.Sharpness(prob=aug_prob),

            # [致命错误修复] 使用 re_size (512) 而非 size (585)
            # 确保生成的噪声掩码与裁剪后的图片尺寸一致
            FundusAug.Halo(re_size, prob=aug_prob),
            FundusAug.Hole(re_size, prob=aug_prob),
            FundusAug.Spot(re_size, prob=aug_prob),

            FundusAug.Blur(prob=aug_prob)
        ]),

        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        normalize
    ])

    # 3. Validation
    tra_test = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(re_size),
        transforms.ToTensor(),
        normalize
    ])

    # 4. Mask Transform
    # [逻辑修复] 移除 RandomCrop，使用 CenterCrop 配合 Resize
    # 这样既能保证尺寸变为 512，又能避免与 Image 的 RandomCrop 不对齐的问题
    # 同时也保证了 ToTensor 被执行
    tra_mask = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(re_size),
        transforms.ToTensor()
    ])

    return tra_train_weak, tra_test, tra_train_strong, tra_mask


def get_normalize(cfg=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if cfg is not None and hasattr(cfg.MODEL, 'FPT') and hasattr(cfg.MODEL.FPT, 'LPM_PATH'):
        try:
            if os.path.exists(cfg.MODEL.FPT.LPM_PATH):
                processor = AutoImageProcessor.from_pretrained(cfg.MODEL.FPT.LPM_PATH)
                if hasattr(processor, 'image_mean'):
                    mean = processor.image_mean if isinstance(processor.image_mean,
                                                              list) else [processor.image_mean] * 3
                if hasattr(processor, 'image_std'):
                    std = processor.image_std if isinstance(processor.image_std, list) else [processor.image_std] * 3
        except:
            pass
    return transforms.Normalize(mean=mean, std=std)