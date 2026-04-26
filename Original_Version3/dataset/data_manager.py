from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from .fundusaug import square_tight_crop
from torchvision import transforms
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch
import numpy as np
from PIL import Image

def get_dataset(args, cfg):
    if cfg.ALGORITHM != 'GDRNet' and cfg.ALGORITHM != 'CASS_GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)
    batch_size = cfg.BATCH_SIZE
    num_worker = cfg.num_workers
    drop_last = getattr(cfg, 'DROP_LAST', True)
    train_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS, target_domains=cfg.DATASET.TARGET_DOMAINS, mode='train', trans_basic=train_ts, trans_mask=tra_fundus)
    train_sampler = None
    shuffle = True
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker, drop_last=drop_last, pin_memory=True, sampler=train_sampler)
    val_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS, target_domains=cfg.DATASET.TARGET_DOMAINS, mode='val', trans_basic=test_ts)
    test_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS, target_domains=cfg.DATASET.TARGET_DOMAINS, mode='test', trans_basic=test_ts)
    val_sampler = None
    test_sampler = None
    if args.local_rank != -1:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, sampler=val_sampler, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, sampler=test_sampler, pin_memory=True)
    dataset_size = [len(train_dataset), len(val_dataset), len(test_dataset)]
    return train_loader, val_loader, test_loader, dataset_size, train_sampler

def get_transform(cfg):
    re_size = 1216
    normalize = get_normalize()
    tra_train = v2.Compose([
        lambda img: square_tight_crop(img, target_size=re_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(45),
        v2.ColorJitter(0.3, 0.3, 0.3, 0.05),
        v2.ToTensor(),
        normalize,
    ])
    tra_test = v2.Compose([
        lambda img: square_tight_crop(img, target_size=re_size),
        v2.ToTensor(),
        normalize,
    ])
    tra_mask = transforms.Compose([transforms.ToTensor()])
    return tra_train, tra_test, tra_mask

def get_pre_FundusAug(cfg):
    jitter_b = getattr(cfg.TRANSFORM, 'COLORJITTER_B', 0.3)
    jitter_c = getattr(cfg.TRANSFORM, 'COLORJITTER_C', 0.3)
    jitter_s = getattr(cfg.TRANSFORM, 'COLORJITTER_S', 0.3)
    jitter_h = getattr(cfg.TRANSFORM, 'COLORJITTER_H', 0.05)
    normalize = get_normalize()

    weak_transforms = v2.Compose([
        lambda img: square_tight_crop(img, target_size=1216),
        v2.Resize((1216, 1216), antialias=True),
        v2.ToTensor(),
        normalize,
    ])

    vit_train_transforms = v2.Compose([
        lambda img: square_tight_crop(img, target_size=1216),
        v2.Resize((1216, 1216), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(45),
        v2.ColorJitter(brightness=jitter_b, contrast=jitter_c, saturation=jitter_s, hue=jitter_h),
        v2.ToTensor(),
        normalize,
    ])

    tra_train = vit_train_transforms
    tra_test = weak_transforms
    tra_mask = transforms.Compose([transforms.ToTensor()])
    return tra_train, tra_test, tra_mask

def get_post_FundusAug(cfg):
    aug_prob = getattr(cfg.TRANSFORM, 'AUGPROB', 0.5)
    size = 1216
    re_size = 1216
    normalize = get_normalize()
    tra_fundus_1 = FundusAug.Compose([FundusAug.Sharpness(prob=aug_prob), FundusAug.Halo(size, prob=aug_prob), FundusAug.Hole(size, prob=aug_prob), FundusAug.Spot(size, prob=aug_prob), FundusAug.Blur(prob=aug_prob)])
    tra_fundus_2 = transforms.Compose([transforms.RandomCrop(re_size), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), normalize])
    return {'post_aug1': tra_fundus_1, 'post_aug2': tra_fundus_2}

def get_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
