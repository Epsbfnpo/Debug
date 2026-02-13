from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from torchvision import transforms
import torchvision.transforms.functional as F  # 引入 functional 用于 padding
from torch.utils.data import DataLoader, DistributedSampler
import torch
import numpy as np
from PIL import Image


# ================= 自定义 Padding 类 =================
class SquarePad:
    """
    将任意尺寸的图片，通过填充黑边（Padding）的方式变成正方形。
    保持原始长宽比，不发生形变。
    """

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        # padding format: (left, top, right, bottom)
        # fill=0 (black)
        return F.pad(image, (p_left, p_top, p_right, p_bottom), 0, 'constant')


# ====================================================

def get_dataset(args, cfg):
    # 预处理选择
    if cfg.ALGORITHM != 'GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)

    batch_size = cfg.BATCH_SIZE
    num_worker = cfg.num_workers
    drop_last = getattr(cfg, 'DROP_LAST', True)

    # --- 训练集 (Source) ---
    train_dataset = GDRBench(
        root=cfg.DATASET.ROOT,
        source_domains=cfg.DATASET.SOURCE_DOMAINS,
        target_domains=cfg.DATASET.TARGET_DOMAINS,
        mode='train',
        trans_basic=train_ts,
        trans_mask=tra_fundus
    )

    train_sampler = None
    shuffle = True
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        drop_last=drop_last,
        pin_memory=True,
        sampler=train_sampler
    )

    # --- 验证/测试集 (Target) ---
    val_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS,
                           target_domains=cfg.DATASET.TARGET_DOMAINS, mode='val', trans_basic=test_ts)

    test_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS,
                            target_domains=cfg.DATASET.TARGET_DOMAINS, mode='test', trans_basic=test_ts)

    # Val/Test DDP Sampler
    val_sampler = None
    test_sampler = None

    if args.local_rank != -1:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        sampler=val_sampler,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        sampler=test_sampler,
        pin_memory=True
    )

    dataset_size = [len(train_dataset), len(val_dataset), len(test_dataset)]

    return train_loader, val_loader, test_loader, dataset_size, train_sampler


def get_transform(cfg):
    # 这个函数是给非 GDRNet 方法用的，既然我们只跑 GDRNet，其实不太重要
    # 但为了兼容性，也可以加上 SquarePad
    size = 256
    re_size = 224
    normalize = get_normalize()
    tra_train = transforms.Compose([
        SquarePad(),  # 先补正
        transforms.Resize((size, size)),  # 再缩放，保证是正方形
        transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        normalize
    ])
    tra_test = transforms.Compose([
        SquarePad(),
        transforms.Resize((size, size)),
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor(),
        normalize
    ])
    tra_mask = transforms.Compose([
        SquarePad(),
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])
    return tra_train, tra_test, tra_mask


def get_pre_FundusAug(cfg):
    """
    针对 GDRNet 的预处理 - 使用 Padding 方案
    """
    jitter_b = getattr(cfg.TRANSFORM, 'COLORJITTER_B', 0.2)
    jitter_c = getattr(cfg.TRANSFORM, 'COLORJITTER_C', 0.2)
    jitter_s = getattr(cfg.TRANSFORM, 'COLORJITTER_S', 0.2)
    jitter_h = getattr(cfg.TRANSFORM, 'COLORJITTER_H', 0.1)
    size = 256
    re_size = 224
    normalize = get_normalize()

    # 1. 训练集
    tra_train = transforms.Compose([
        SquarePad(),  # [核心] 先把非正方形图片补黑边变成正方形
        transforms.Resize((size, size)),  # 然后安全地缩放到 256x256，此时不会发生形变
        transforms.ColorJitter(brightness=jitter_b, contrast=jitter_c, saturation=jitter_s, hue=jitter_h),
        transforms.ToTensor()
    ])

    # 2. 测试集
    tra_test = transforms.Compose([
        SquarePad(),  # 测试集也要 Pad，保证和训练集看到的一致
        transforms.Resize((size, size)),
        transforms.CenterCrop(re_size),
        transforms.ToTensor(),
        normalize
    ])

    # 3. Mask (掩膜)
    # 必须对 Mask 做完全一样的 Pad 操作，否则 Mask 和 Image 就不对齐了！
    tra_mask = transforms.Compose([
        SquarePad(),
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    return tra_train, tra_test, tra_mask


def get_post_FundusAug(cfg):
    aug_prob = getattr(cfg.TRANSFORM, 'AUGPROB', 0.5)
    size = 256
    re_size = 224
    normalize = get_normalize()
    tra_fundus_1 = FundusAug.Compose(
        [FundusAug.Sharpness(prob=aug_prob), FundusAug.Halo(size, prob=aug_prob), FundusAug.Hole(size, prob=aug_prob),
         FundusAug.Spot(size, prob=aug_prob), FundusAug.Blur(prob=aug_prob)])
    tra_fundus_2 = transforms.Compose(
        [transforms.RandomCrop(re_size), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), normalize])
    return {'post_aug1': tra_fundus_1, 'post_aug2': tra_fundus_2}


def get_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])