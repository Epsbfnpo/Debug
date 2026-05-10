from .GDRBench import GDRBench
from . import fundusaug as FundusAug
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch
import numpy as np
import random
from PIL import Image

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        return F.pad(image, (p_left, p_top, p_right, p_bottom), 0, 'constant')

def get_dataset(args, cfg):
    if cfg.ALGORITHM != 'GDRNet' and cfg.ALGORITHM != 'CASS_GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)
    batch_size = cfg.BATCH_SIZE
    num_worker = cfg.num_workers
    drop_last = getattr(cfg, 'DROP_LAST', True)
    train_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS, target_domains=cfg.DATASET.TARGET_DOMAINS, mode='train', trans_basic=train_ts, trans_mask=tra_fundus)
    g_train = torch.Generator()
    g_train.manual_seed(cfg.SEED)
    g_val = torch.Generator()
    g_val.manual_seed(cfg.SEED + 1)
    g_test = torch.Generator()
    g_test.manual_seed(cfg.SEED + 2)
    train_sampler = None
    shuffle = True
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=cfg.SEED)
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker, drop_last=drop_last, pin_memory=True, sampler=train_sampler, worker_init_fn=seed_worker, generator=g_train)
    val_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS, target_domains=cfg.DATASET.TARGET_DOMAINS, mode='val', trans_basic=test_ts)
    test_dataset = GDRBench(root=cfg.DATASET.ROOT, source_domains=cfg.DATASET.SOURCE_DOMAINS, target_domains=cfg.DATASET.TARGET_DOMAINS, mode='test', trans_basic=test_ts)
    val_sampler = None
    test_sampler = None
    if args.local_rank != -1:
        val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=cfg.SEED)
        test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=cfg.SEED)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, sampler=val_sampler, pin_memory=True, worker_init_fn=seed_worker, generator=g_val)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, sampler=test_sampler, pin_memory=True, worker_init_fn=seed_worker, generator=g_test)
    dataset_size = [len(train_dataset), len(val_dataset), len(test_dataset)]
    return train_loader, val_loader, test_loader, dataset_size, train_sampler

def get_transform(cfg):
    size = getattr(cfg.TRANSFORM, 'INPUT_SIZE', 224)
    re_size = getattr(cfg.TRANSFORM, 'CROP_SIZE', size)
    normalize = get_normalize()
    tra_train = transforms.Compose([SquarePad(), transforms.Resize((size, size)), transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), transforms.RandomGrayscale(), transforms.ToTensor(), normalize])
    tra_test = transforms.Compose([SquarePad(), transforms.Resize((size, size)), transforms.Resize((re_size, re_size)), transforms.ToTensor(), normalize])
    tra_mask = transforms.Compose([SquarePad(), transforms.Resize((re_size, re_size)), transforms.ToTensor()])
    return tra_train, tra_test, tra_mask

def get_pre_FundusAug(cfg):
    jitter_b = getattr(cfg.TRANSFORM, 'COLORJITTER_B', 0.2)
    jitter_c = getattr(cfg.TRANSFORM, 'COLORJITTER_C', 0.2)
    jitter_s = getattr(cfg.TRANSFORM, 'COLORJITTER_S', 0.2)
    jitter_h = getattr(cfg.TRANSFORM, 'COLORJITTER_H', 0.1)
    size = getattr(cfg.TRANSFORM, 'INPUT_SIZE', 224)
    re_size = getattr(cfg.TRANSFORM, 'CROP_SIZE', size)
    normalize = get_normalize()
    tra_train = transforms.Compose([SquarePad(), transforms.Resize((size, size)), transforms.ColorJitter(brightness=jitter_b, contrast=jitter_c, saturation=jitter_s, hue=jitter_h), transforms.ToTensor()])
    tra_test = transforms.Compose([SquarePad(), transforms.Resize((size, size)), transforms.CenterCrop(re_size), transforms.ToTensor(), normalize])
    tra_mask = transforms.Compose([SquarePad(), transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])
    return tra_train, tra_test, tra_mask

def get_post_FundusAug(cfg):
    aug_prob = getattr(cfg.TRANSFORM, 'AUGPROB', 0.5)
    size = getattr(cfg.TRANSFORM, 'INPUT_SIZE', 224)
    re_size = getattr(cfg.TRANSFORM, 'CROP_SIZE', size)
    normalize = get_normalize()
    tra_fundus_1 = FundusAug.Compose([FundusAug.Sharpness(prob=aug_prob), FundusAug.Halo(size, prob=aug_prob), FundusAug.Hole(size, prob=aug_prob), FundusAug.Spot(size, prob=aug_prob), FundusAug.Blur(prob=aug_prob)])
    tra_fundus_2 = transforms.Compose([transforms.RandomCrop(re_size), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), normalize])
    return {'post_aug1': tra_fundus_1, 'post_aug2': tra_fundus_2}

def get_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
