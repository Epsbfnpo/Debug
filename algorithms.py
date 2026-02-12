import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, collections, logging
from collections import OrderedDict
from copy import deepcopy
from itertools import combinations
from torch.cuda.amp import autocast

import utils.misc as misc
from utils.validate import algorithm_validate
import modeling.model_manager as models
from modeling.losses import DahLoss, DahLoss_Siam_Fastmoco_v0
from modeling.nets import LossValley, AveragedModel
from dataset.data_manager import get_pre_FundusAug

from backpack import backpack, extend
from backpack.extensions import BatchGrad

ALGORITHMS = ['ERM', 'GDRNet', 'GREEN', 'CABNet', 'MixupNet', 'MixStyleNet', 'Fishr', 'DRGen']


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    def __init__(self, num_classes, cfg):
        super(Algorithm, self).__init__()
        self.cfg = cfg
        self.epoch = 0

    def update(self, minibatches):
        raise NotImplementedError

    def update_epoch(self, epoch):
        self.epoch = epoch
        return epoch

    def validate(self, val_loader, test_loader, writer):
        val_metrics = algorithm_validate(self, val_loader, writer, self.epoch, 'val')

        if test_loader is not None:
            test_metrics = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
        else:
            test_metrics = {'auc': 0.0, 'acc': 0.0, 'f1': 0.0, 'qwk': 0.0, 'loss': 0.0}

        return val_metrics, test_metrics

    def save_model(self, log_path):
        raise NotImplementedError

    def renew_model(self, log_path):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def _get_net_state_dict(self):
        if hasattr(self.network, 'module'):
            return self.network.module.state_dict()
        return self.network.state_dict()

    def _load_net_state_dict(self, state_dict):
        is_ddp = hasattr(self.network, 'module')

        ckpt_keys = list(state_dict.keys())
        has_module_prefix = any(k.startswith('module.') for k in ckpt_keys)

        if is_ddp and not has_module_prefix:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict['module.' + k] = v
            self.network.load_state_dict(new_state_dict)

        elif not is_ddp and has_module_prefix:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.network.load_state_dict(new_state_dict)

        else:
            self.network.load_state_dict(state_dict)


class ERM(Algorithm):
    def __init__(self, num_classes, cfg):
        super(ERM, self).__init__(num_classes, cfg)

        self.network = models.get_net(cfg)
        self.classifier = models.get_classifier(self.network.out_features(), cfg)

        if 'FPT' in cfg.BACKBONE:
            print(f">> [Optimizer Info] Detected {cfg.BACKBONE}, switching to AdamW optimizer.")
            self.optimizer = torch.optim.AdamW(
                [{"params": self.network.parameters()}, {"params": self.classifier.parameters()}], lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY)
        else:
            self.optimizer = torch.optim.SGD(
                [{"params": self.network.parameters()}, {"params": self.classifier.parameters()}], lr=cfg.LEARNING_RATE,
                momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY, nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        features = self.network(image)
        output = self.classifier(features)
        loss = F.cross_entropy(output, label)

        loss.backward()
        self.optimizer.step()

        return {'loss': loss}

    def save_model(self, log_path):
        logging.info("Saving best model...")
        net_state = self._get_net_state_dict()
        torch.save(net_state, os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))

    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')

        self._load_net_state_dict(torch.load(net_path, map_location='cpu'))
        self.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))

    def predict(self, x):
        return self.classifier(self.network(x))


class GDRNet(Algorithm):
    """
    GDRNet (MASK_SIAM v0 完全体) + FPT+ + DINOv3-FD + SPMix
    """

    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)
        self.cfg = cfg

        # 1. Network Backbone (FPT+)
        self.network = models.get_net(cfg)

        # MASK_SIAM v0 参数
        dim_in = 2048
        feat_dim = 512

        # FastMoCo 参数
        self.split_num = 2
        self.combs = 3

        # 2. Projector & Predictor
        self.projector = nn.Sequential(
            nn.Linear(dim_in, feat_dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feat_dim, dim_in, bias=False)
        )
        self.predictor = nn.Sequential(
            nn.Linear(dim_in, dim_in, bias=False)
        )
        self._init_weights(self.projector)
        self._init_weights(self.predictor)

        # 3. Classifier
        self.classifier = models.get_classifier(dim_in, cfg)

        # 4. EMA Teacher
        self.network_ema = deepcopy(self.network)
        self.classifier_ema = deepcopy(self.classifier)
        for p in self.network_ema.parameters(): p.requires_grad = False
        for p in self.classifier_ema.parameters(): p.requires_grad = False

        # 5. Memory Queue
        self.K = 1024
        self.register_buffer("queue", torch.randn(self.K, dim_in))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(self.K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.num_positive = getattr(cfg, 'POSITIVE', 4)

        # 6. Optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.network.parameters(), 'fix_lr': False},
                {"params": self.classifier.parameters(), 'fix_lr': False},
                {"params": self.projector.parameters(), 'fix_lr': False},
                {"params": self.predictor.parameters(), 'fix_lr': True},
            ],
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )

        # 7. Loss
        self.criterion = DahLoss_Siam_Fastmoco_v0(
            cfg=cfg,
            max_iteration=cfg.EPOCHS,
            training_domains=cfg.DATASET.SOURCE_DOMAINS,
            beta=cfg.GDRNET.BETA,
            temperature=cfg.GDRNET.TEMPERATURE,
            scaling_factor=cfg.GDRNET.SCALING_FACTOR,
            fastmoco=1.0
        )
        self.E_dis = nn.MSELoss()

        # [关键修改] reduction='none' 允许我们获得每个样本的 Loss，从而乘以对应的 lam
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def change_random_matrix(self):
        # MASK_SIAM 原版逻辑：生成一个 (Feature_Dim, Projection_Dim) 的随机矩阵
        # 通常 ResNet50 的特征维度是 2048，投影维度设为 512
        # .cuda() 确保矩阵在 GPU 上，与特征计算保持一致
        self.random_matrix = torch.randn(2048, 512).cuda()

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0)

    # --- MASK_SIAM Core Logic: Queue Management ---
    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        # Replace the queue at ptr (wrap around)
        if ptr + batch_size > self.K:
            rem = self.K - ptr
            self.queue[ptr:self.K] = features[:rem]
            self.queue_labels[ptr:self.K] = labels[:rem]
            self.queue[0:batch_size - rem] = features[rem:]
            self.queue_labels[0:batch_size - rem] = labels[rem:]
        else:
            self.queue[ptr: ptr + batch_size] = features
            self.queue_labels[ptr: ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def sample_target(self, features, labels):
        neighbor = []
        for i, label in enumerate(labels):
            pos = torch.where(self.queue_labels == label)[0]
            if len(pos) != 0:
                choice = torch.multinomial(torch.ones_like(pos).type(torch.FloatTensor), self.num_positive,
                                           replacement=True)
                idx = pos[choice]
                neighbor.append(self.queue[idx].mean(0))
            else:
                neighbor.append(features[i])
        neighbor = torch.stack(neighbor, dim=0)
        return neighbor

    # --- MASK_SIAM Core Logic: Patchify & Masking ---
    def patchify(self, imgs, block=32):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = block
        # Ensure divisible
        if imgs.shape[2] % p != 0:
            new_size = (imgs.shape[2] // p) * p
            imgs = F.interpolate(imgs, size=(new_size, new_size))

        h = w = imgs.shape[2] // p
        n = imgs.shape[0]

        x = imgs.reshape(shape=(n, 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x, block=32):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = block
        h = w = int(x.shape[1] ** .5)
        n = x.shape[0]

        x = x.reshape(shape=(n, h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(n, 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand([N, L], device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Return masks extended to feature dim for multiplication
        return mask.unsqueeze(-1), (1 - mask.unsqueeze(-1))

    # --- MASK_SIAM Core Logic: FastMoCo Splits ---
    def _local_split_and_resize(self, x):
        """
        1. 将图像 x (N, C, H, W) 空间切分为 split_num*split_num 份
        2. 将每一份 Resize 回 (H, W)，以便输入 FPT+
        """
        B, C, H, W = x.shape
        # 计算切片大小
        h_step = H // self.split_num
        w_step = W // self.split_num

        crops = []
        for i in range(self.split_num):
            for j in range(self.split_num):
                # 切片
                h_start, h_end = i * h_step, (i + 1) * h_step
                w_start, w_end = j * w_step, (j + 1) * w_step
                crop = x[:, :, h_start:h_end, w_start:w_end]

                # [核心适配] Resize 回原始大小，适配 FPT+ 的 Position Embedding
                crop_resized = F.interpolate(crop, size=(H, W), mode='bilinear', align_corners=False)
                crops.append(crop_resized)

        # 堆叠成一个大的 Batch: (4*B, C, H, W)
        x_splits = torch.cat(crops, dim=0)
        return x_splits

    def update_ema_model(self, momentum=0.996):
        """Update EMA target model parameters."""
        with torch.no_grad():
            for param_q, param_k in zip(self.network.parameters(), self.network_ema.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
            for param_q, param_k in zip(self.classifier.parameters(), self.classifier_ema.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def update(self, minibatch, step=None, accum_iter=None):
        self.change_random_matrix()

        # [Critical Fix] 解包逻辑与 GDRBench 返回值严格对齐
        all_data = minibatch
        img_weak = all_data[0].cuda()
        img_strong = all_data[1].cuda()

        # 检查是否包含 Mixed 数据
        img_mixed = None
        if len(all_data) >= 9:
            img_mixed = all_data[2].cuda()
            mask = all_data[3].cuda()
            label = all_data[4].cuda()
            label_mixed_b = all_data[5].cuda()
            lam = all_data[6]  # Tensor Shape: (Batch_Size,)
            domain = all_data[7].cuda()
        else:
            mask = all_data[3].cuda() if len(all_data) > 3 else None
            label = all_data[4].cuda()
            domain = all_data[7].cuda()

        self.optimizer.zero_grad()

        # ==========================================
        # 1. MASK_SIAM 核心流程
        # ==========================================

        # A. Mask 生成
        with torch.no_grad():
            bs = self.cfg.BLOCK_SIZE
            mask_ratio = self.cfg.MASK_RATIO
            img_new_patches = self.patchify(img_strong, block=bs)
            img_ori_patches = self.patchify(img_weak, block=bs)
            mask_m, mask_c = self.random_masking(img_new_patches, mask_ratio)
            img_new_masked = self.unpatchify(img_new_patches * mask_m, block=bs)
            img_ori_masked_c = self.unpatchify(img_ori_patches * mask_c, block=bs)

        # B. 正常视图前向传播
        feat_ori = self.network(img_weak)['features']

        # [优化点 1] 只需要跑一次 Forward，保存完整 Output 供后面 DINOv3-FD 使用
        out_new_full = self.network(img_strong)
        feat_new = out_new_full['features']

        # C. 掩码视图前向传播
        out_new_masked = self.network(img_new_masked)
        feat_new_masked = out_new_masked['features']
        feat_new_masked_list = out_new_masked['features_list']

        feat_ori_masked_c = self.network(img_ori_masked_c)['features']

        # D. Logits
        logits_new = self.classifier(feat_new)
        logits_ori = self.classifier(feat_ori)
        logits_new_masked = self.classifier(feat_new_masked)
        logits_ori_masked_c = self.classifier(feat_ori_masked_c)

        # E. Teacher (EMA)
        with torch.no_grad():
            out_ema = self.network_ema(img_strong)
            feat_ema_list = out_ema['features_list']

        # F. Projector & Predictor
        z1 = self.projector(feat_ori)
        z2 = self.projector(feat_new)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        z_new_masked = self.projector(feat_new_masked)
        z_ori_masked_c = self.projector(feat_ori_masked_c)
        p_new_masked = self.predictor(z_new_masked)
        p_ori_masked_c = self.predictor(z_ori_masked_c)

        # G. Supervised Targets
        z1_sup = self.sample_target(z1.detach(), label)
        z2_sup = self.sample_target(z2.detach(), label)

        # ==========================================
        # 2. FastMoCo Split 逻辑
        # ==========================================
        img_splits_weak = self._local_split_and_resize(img_weak)
        img_splits_strong = self._local_split_and_resize(img_strong)

        feat_splits_weak = self.network(img_splits_weak)['features']
        feat_splits_strong = self.network(img_splits_strong)['features']

        feat_list_weak = list(feat_splits_weak.split(img_weak.size(0), dim=0))
        feat_list_strong = list(feat_splits_strong.split(img_strong.size(0), dim=0))

        z1_orthmix_feat = torch.cat(
            list(map(lambda x: sum(x) / self.combs, list(combinations(feat_list_weak, r=self.combs)))), dim=0
        )
        z2_orthmix_feat = torch.cat(
            list(map(lambda x: sum(x) / self.combs, list(combinations(feat_list_strong, r=self.combs)))), dim=0
        )

        z1_orthmix = self.projector(z1_orthmix_feat)
        z2_orthmix = self.projector(z2_orthmix_feat)
        p1_orthmix = self.predictor(z1_orthmix)
        p2_orthmix = self.predictor(z2_orthmix)

        p1_orthmix_list = list(p1_orthmix.split(img_weak.size(0), dim=0))
        p2_orthmix_list = list(p2_orthmix.split(img_strong.size(0), dim=0))

        # ==========================================
        # 3. SPMix Loss (独立流)
        # ==========================================
        loss_spmix = torch.tensor(0.0).cuda()  # 初始化为 Tensor 防止报错
        if img_mixed is not None:
            logits_mixed = self.classifier(self.network(img_mixed)['features'])

            # [修改点 2] 显式计算 per-sample loss，然后求 mean 得到标量
            # 这里的 ce_loss 必须是 reduction='none' (在 __init__ 中设置)
            loss_per_sample = lam * self.ce_loss(logits_mixed, label) + (1 - lam) * self.ce_loss(logits_mixed,
                                                                                                 label_mixed_b)
            loss_spmix = loss_per_sample.mean()

        # ==========================================
        # 4. 计算总 Loss
        # ==========================================
        self.dequeue_and_enqueue(z2.detach(), label)

        loss_main, loss_dict = self.criterion(
            [logits_new, logits_ori, logits_new_masked, logits_ori_masked_c],
            [
                feat_ori, feat_new, z1, z2, z1_sup, z2_sup, p1, p2,
                z1_orthmix, z2_orthmix, p1_orthmix_list, p2_orthmix_list
            ],
            [z_new_masked, z_ori_masked_c, p_new_masked, p_ori_masked_c],
            label, domain, random_matrix=self.random_matrix
        )

        loss_kd = 0.0
        temp_kd = getattr(self.cfg, 'KD', 1.0)
        if feat_new_masked_list and feat_ema_list:
            min_len = min(len(feat_new_masked_list), len(feat_ema_list))
            for i in range(min_len):
                loss_kd += self.E_dis(feat_new_masked_list[i], feat_ema_list[i].detach())

        # DINOv3-FD Orthogonality Loss
        # [优化点 1 的后续] 直接使用之前保存的 out_new_full，不再重复计算
        tra_feat = out_new_full.get('tra_feat', None)
        dino_raw = out_new_full.get('dino_raw', None)
        loss_ortho = 0.0
        if tra_feat is not None and dino_raw is not None:
            u = tra_feat.mean(dim=1) if tra_feat.dim() == 3 else tra_feat
            v = dino_raw.detach().mean(dim=1) if dino_raw.dim() == 3 else dino_raw.detach()
            cosine = F.cosine_similarity(u, v, dim=-1)
            loss_ortho = (cosine ** 2).mean()

        total_loss = loss_main + temp_kd * loss_kd + 0.1 * loss_ortho + 1.0 * loss_spmix

        # 确保日志记录的是标量
        loss_dict['loss_spmix'] = loss_spmix.item()
        loss_dict['loss_ortho'] = loss_ortho.item() if isinstance(loss_ortho, torch.Tensor) else 0

        total_loss.backward()
        self.optimizer.step()

        self.update_ema_model()
        self.criterion.update_alpha(self.epoch)

        return loss_dict

    def save_model(self, log_path):
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
        torch.save(self.projector.state_dict(), os.path.join(log_path, 'best_projector.pth'))
        torch.save(self.predictor.state_dict(), os.path.join(log_path, 'best_predictor.pth'))

    def renew_model(self, log_path):
        self.network.load_state_dict(torch.load(os.path.join(log_path, 'best_model.pth')))
        self.classifier.load_state_dict(torch.load(os.path.join(log_path, 'best_classifier.pth')))

    def predict(self, x):
        feat = self.network(x)['features']
        return self.classifier(feat)


class GREEN(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GREEN, self).__init__(num_classes, cfg)
        self.network = models.get_net(cfg)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
            nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        output = self.network(image)
        loss = F.cross_entropy(output, label)

        loss.backward()
        self.optimizer.step()
        return {'loss': loss}

    def save_model(self, log_path):
        logging.info("Saving best model...")
        net_state = self._get_net_state_dict()
        torch.save(net_state, os.path.join(log_path, 'best_model.pth'))

    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        self._load_net_state_dict(torch.load(net_path, map_location='cpu'))

    def predict(self, x):
        return self.network(x)


class CABNet(ERM):
    def __init__(self, num_classes, cfg):
        super(CABNet, self).__init__(num_classes, cfg)


class MixStyleNet(ERM):
    def __init__(self, num_classes, cfg):
        super(MixStyleNet, self).__init__(num_classes, cfg)


class MixupNet(ERM):
    def __init__(self, num_classes, cfg):
        super(MixupNet, self).__init__(num_classes, cfg)
        self.criterion_CE = torch.nn.CrossEntropyLoss()

    def update(self, minibatch, env_feats=None):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        inputs, targets_a, targets_b, lam = self.mixup_data(image, label)
        outputs = self.predict(inputs)
        loss = self.mixup_criterion(self.criterion_CE, outputs, targets_a, targets_b, lam)

        loss.backward()
        self.optimizer.step()

        return {'loss': loss}

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Fishr(ERM):
    def __init__(self, num_classes, cfg):
        super(Fishr, self).__init__(num_classes, cfg)
        self.num_groups = cfg.FISHR.NUM_GROUPS
        self.network = models.get_net(cfg)
        self.classifier = extend(models.get_classifier(self.network._out_features, cfg))
        self.optimizer = None
        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [misc.MovingAverage(cfg.FISHR.EMA, oneminusema_correction=True) for _ in
                               range(self.num_groups)]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            list(self.network.parameters()) + list(self.classifier.parameters()),
            lr=self.cfg.LEARNING_RATE,
            momentum=self.cfg.MOMENTUM,
            weight_decay=self.cfg.WEIGHT_DECAY,
            nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        all_x = image
        all_y = label
        len_minibatches = [image.shape[0]]

        all_z = self.network(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.cfg.FISHR.PENALTY_ANNEAL_ITERS:
            penalty_weight = self.cfg.FISHR.LAMBDA
            if self.update_count == self.cfg.FISHR.PENALTY_ANNEAL_ITERS != 0:
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True)
        dict_grads = OrderedDict(
            [(name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)) for name, weights in
             self.classifier.named_parameters()])
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        grads_var_per_domain = [{} for _ in range(self.num_groups)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)
        for domain_id in range(self.num_groups):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(grads_var_per_domain[domain_id])
        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):
        grads_var = OrderedDict([(name, torch.stack(
            [grads_var_per_domain[domain_id][name] for domain_id in range(self.num_groups)], dim=0).mean(dim=0)) for
                                 name in grads_var_per_domain[0].keys()])
        penalty = 0
        for domain_id in range(self.num_groups):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_groups

    def l2_between_dicts(self, dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (torch.cat(tuple([t.view(-1) for t in dict_1_values])) - torch.cat(
            tuple([t.view(-1) for t in dict_2_values]))).pow(2).mean()


class DRGen(Algorithm):
    def __init__(self, num_classes, cfg):
        super(DRGen, self).__init__(num_classes, cfg)
        algorithm_class = get_algorithm_class('Fishr')
        self.algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
        self.optimizer = self.algorithm.optimizer

        self.swad_algorithm = AveragedModel(self.algorithm)
        self.swad_algorithm.cuda()
        self.swad = LossValley(None, cfg.DRGEN.N_CONVERGENCE, cfg.DRGEN.N_TOLERANCE, cfg.DRGEN.TOLERANCE_RATIO)

    def update(self, minibatch):
        loss_dict_iter = self.algorithm.update(minibatch)
        if self.swad:
            self.swad_algorithm.update_parameters(self.algorithm, step=self.epoch)
        return loss_dict_iter

    def validate(self, val_loader, test_loader, writer):
        swad_val_auc = -1
        swad_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate(self.algorithm, val_loader, writer, self.epoch, 'val(Fishr)')
            test_auc, test_loss = algorithm_validate(self.algorithm, test_loader, writer, self.epoch, 'test(Fishr)')

            if self.swad:
                def prt_results_fn(results):
                    print(results)

                self.swad.update_and_evaluate(self.swad_algorithm, val_auc, val_loss, prt_results_fn)

                if self.epoch != self.cfg.EPOCHS:
                    self.swad_algorithm = self.swad.get_final_model()
                    swad_val_auc, swad_val_loss = algorithm_validate(self.swad_algorithm, val_loader, writer,
                                                                     self.epoch, 'val')
                    swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.epoch,
                                                             'test')

                    if hasattr(self.swad, "dead_valley") and self.swad.dead_valley:
                        logging.info("SWAD valley is dead -> not stop !")

                    self.swad_algorithm = AveragedModel(self.algorithm)  # reset

            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1

        else:
            self.swad_algorithm = self.swad.get_final_model()
            logging.warning("Evaluate SWAD ...")
            swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer,
                                                     self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('(last) swad test auc: {}  loss: {}'.format(swad_auc, swad_loss))

        return swad_val_auc, swad_auc

    def save_model(self, log_path):
        self.algorithm.save_model(log_path)

    def renew_model(self, log_path):
        self.algorithm.renew_model(log_path)

    def predict(self, x):
        return self.swad_algorithm.predict(x)