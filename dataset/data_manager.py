from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch
from transformers import AutoImageProcessor
import os


def get_dataset(args, cfg):
    if cfg.ALGORITHM != 'GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)

    batch_size = cfg.BATCH_SIZE
    drop_last = getattr(cfg, 'DROP_LAST', True)

    num_worker = getattr(cfg, 'WORKERS', 4)

    train_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS, target_domains=cfg.DATASET.TARGET_DOMAINS, mode='train', trans_basic=train_ts, trans_mask=tra_fundus)

    train_sampler = None
    train_shuffle = True
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_shuffle = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_worker, drop_last=drop_last, pin_memory=True, sampler=train_sampler)

    val_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS, target_domains=cfg.DATASET.TARGET_DOMAINS, mode='val', trans_basic=test_ts)

    val_sampler = None
    if args.local_rank != -1:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True, sampler=val_sampler)

    test_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS, target_domains=cfg.DATASET.TARGET_DOMAINS, mode='test', trans_basic=test_ts)

    test_sampler = None
    if args.local_rank != -1:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True, sampler=test_sampler)

    dataset_size = [len(train_dataset), len(val_dataset), len(test_dataset)]

    return train_loader, val_loader, test_loader, dataset_size, train_sampler


def get_transform(cfg):
    re_size = getattr(cfg.INPUT, 'SIZE', 224)

    normalize = get_normalize(cfg)

    tra_train = transforms.Compose([transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.RandomGrayscale(), transforms.ToTensor(), normalize])

    tra_test = transforms.Compose([transforms.Resize((re_size, re_size)), transforms.ToTensor(), normalize])

    tra_mask = transforms.Compose([transforms.Resize((re_size, re_size)), transforms.ToTensor()])

    return tra_train, tra_test, tra_mask


def get_pre_FundusAug(cfg):
    jitter_b = getattr(cfg.TRANSFORM, 'COLORJITTER_B', 0.2)
    jitter_c = getattr(cfg.TRANSFORM, 'COLORJITTER_C', 0.2)
    jitter_s = getattr(cfg.TRANSFORM, 'COLORJITTER_S', 0.2)
    jitter_h = getattr(cfg.TRANSFORM, 'COLORJITTER_H', 0.1)

    re_size = getattr(cfg.INPUT, 'SIZE', 224)

    size = int(re_size * (256 / 224))

    normalize = get_normalize(cfg)

    tra_train = transforms.Compose([transforms.Resize((size, size)), transforms.ColorJitter(brightness=jitter_b, contrast=jitter_c, saturation=jitter_s, hue=jitter_h), transforms.ToTensor()])

    tra_test = transforms.Compose([transforms.Resize((size, size)), transforms.CenterCrop(re_size), transforms.ToTensor(), normalize])

    tra_mask = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    return tra_train, tra_test, tra_mask


def get_post_FundusAug(cfg):
    aug_prob = getattr(cfg.TRANSFORM, 'AUGPROB', 0.5)

    re_size = getattr(cfg.INPUT, 'SIZE', 224)
    size = int(re_size * (256 / 224))

    normalize = get_normalize(cfg)

    tra_fundus_1 = FundusAug.Compose([FundusAug.Sharpness(prob=aug_prob), FundusAug.Halo(size, prob=aug_prob), FundusAug.Hole(size, prob=aug_prob), FundusAug.Spot(size, prob=aug_prob), FundusAug.Blur(prob=aug_prob)])

    tra_fundus_2 = transforms.Compose([transforms.RandomCrop(re_size), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), normalize])

    return {'post_aug1': tra_fundus_1, 'post_aug2': tra_fundus_2}


def get_normalize(cfg=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if cfg is not None and hasattr(cfg.MODEL, 'FPT') and hasattr(cfg.MODEL.FPT, 'LPM_PATH'):
        try:
            processor = AutoImageProcessor.from_pretrained(cfg.MODEL.FPT.LPM_PATH)
            if hasattr(processor, 'image_mean') and hasattr(processor, 'image_std'):
                if isinstance(processor.image_mean, (list, tuple)):
                    mean = processor.image_mean
                elif isinstance(processor.image_mean, (float, int)):
                    mean = [processor.image_mean] * 3

                if isinstance(processor.image_std, (list, tuple)):
                    std = processor.image_std
                elif isinstance(processor.image_std, (float, int)):
                    std = [processor.image_std] * 3
        except Exception:
            pass

    return transforms.Normalize(mean=mean, std=std)