import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, collections, logging
from collections import OrderedDict
from copy import deepcopy
from itertools import combinations

import utils.misc as misc
from utils.validate import algorithm_validate
import modeling.model_manager as models
from modeling.losses import DahLoss, DahLoss_Siam_Fastmoco_v0
from modeling.nets import LossValley, AveragedModel
from dataset.data_manager import get_post_FundusAug
# from utils.sdclr_utils import Mask

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
        val_auc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
        test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
        return val_auc, test_auc

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
            self.optimizer = torch.optim.AdamW([{"params":self.network.parameters()}, {"params":self.classifier.parameters()}], lr = cfg.LEARNING_RATE, weight_decay = cfg.WEIGHT_DECAY)
        else:
            self.optimizer = torch.optim.SGD([{"params":self.network.parameters()}, {"params":self.classifier.parameters()}], lr = cfg.LEARNING_RATE, momentum = cfg.MOMENTUM, weight_decay = cfg.WEIGHT_DECAY, nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        features = self.network(image)
        output = self.classifier(features)
        loss = F.cross_entropy(output, label)

        loss.backward()
        self.optimizer.step()

        return {'loss':loss}

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


class GDRNet(ERM):
    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)

        # 1. 主干网络 (FPTPlusNet, 内含 SideNet 和 LoRA-ViT)
        # 注意：现在 nets.py 修改后，这里面包含了可训练的 LoRA 参数
        self.network = models.get_net(cfg)

        # 2. EMA Teacher
        self.network_ema = deepcopy(self.network)
        for param in self.network_ema.parameters():
            param.requires_grad = False

        dim_in = 2048  # CNN特征维度
        feat_dim = 512  # 投影维度
        dino_dim = 768  # ViT特征维度

        # [CNN] Projector & Predictor
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim, bias=False), nn.BatchNorm1d(feat_dim, affine=False)
        )
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        # ----------------------------------------------------------------------
        # [DINOv3-FD 全套组件] Feature Disentanglement
        # ----------------------------------------------------------------------
        print(f">> [GDRNet] Initializing DINOv3-FD Full Suite (Disentanglement)...")

        # 1. TRA Head (Task-Relevant Adapter): 负责提取和疾病相关的特征
        self.tra_projector = nn.Sequential(
            nn.Linear(dino_dim, dino_dim // 2),
            nn.LayerNorm(dino_dim // 2),
            nn.GELU(),
            nn.Linear(dino_dim // 2, dino_dim)
        )

        # 2. TIA Head (Task-Irrelevant Adapter): 负责提取背景、噪声等特征
        self.tia_projector = nn.Sequential(
            nn.Linear(dino_dim, dino_dim // 2),
            nn.LayerNorm(dino_dim // 2),
            nn.GELU(),
            nn.Linear(dino_dim // 2, dino_dim)
        )

        # 3. TRA Classifier (Task Loss): 只有 TRA 特征被允许用来做分类
        self.dino_classifier = nn.Linear(dino_dim, num_classes)

        # 4. DINO Projector (Distillation):
        # 让 CNN 模仿 TRA 特征
        self.dino_projector = nn.Sequential(
            nn.Linear(dino_dim, dim_in, bias=False), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim, bias=False), nn.BatchNorm1d(feat_dim, affine=False)
        )
        # ----------------------------------------------------------------------

        # 优化器：必须包含所有新组件和 LoRA 参数
        # 这里的 self.network.parameters() 包含了 nets.py 里的 SideResNet 和 FrozenViT(中的LoRA)
        self.optimizer = torch.optim.Adam([
            {"params": self.network.parameters(), 'fix_lr': False},
            {"params": self.projector.parameters(), 'fix_lr': False},
            {"params": self.predictor.parameters(), 'fix_lr': True},

            # FD 组件
            {"params": self.tra_projector.parameters(), 'fix_lr': False},
            {"params": self.tia_projector.parameters(), 'fix_lr': False},
            {"params": self.dino_classifier.parameters(), 'fix_lr': False},
            {"params": self.dino_projector.parameters(), 'fix_lr': False},
        ], lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS, eta_min=0.00015
        )

        # MoCo Queue
        self.K = cfg.MOCO_QUEUE_K
        self.register_buffer("queue", torch.randn(self.K, feat_dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(self.K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.num_positive = cfg.POSITIVE if hasattr(cfg, 'POSITIVE') else 0

        self.criterion = DahLoss_Siam_Fastmoco_v0(
            cfg=cfg, beta=cfg.GDRNET.BETA, max_iteration=cfg.EPOCHS,
            training_domains=cfg.DATASET.SOURCE_DOMAINS,
            temperature=cfg.GDRNET.TEMPERATURE,
            scaling_factor=cfg.GDRNET.SCALING_FACTOR, fastmoco=True
        )

        self.fundusAug = get_post_FundusAug(cfg)
        self.split_num = 2
        self.combs = 3
        self.global_step = 0
        self.sdclr_enabled = False  # 彻底关闭 SDCLR

    def update_epoch(self, epoch):
        self.epoch = epoch
        if self.scheduler is not None:
            self.scheduler.step()
        return epoch

    def patchify(self, imgs, block=32):
        p = block
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        n = imgs.shape[0]
        x = imgs.reshape(shape=(n, 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x, block=32):
        p = block
        h = w = int(x.shape[1] ** .5)
        n = x.shape[0]
        assert h * w == x.shape[1]
        x = x.reshape(shape=(n, h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(n, 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand([N, L], device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        x_masked = x * mask.unsqueeze(-1)
        return mask.unsqueeze(-1), (1 - mask.unsqueeze(-1))

    def _local_split(self, x):
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)
        x = torch.cat(xs, dim=0)
        return x

    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            rem = self.K - ptr
            self.queue[ptr: self.K, :] = features[:rem, :]
            self.queue_labels[ptr: self.K] = labels[:rem]
            self.queue[0: batch_size - rem, :] = features[rem:, :]
            self.queue_labels[0: batch_size - rem] = labels[rem:]
        else:
            self.queue[ptr: ptr + batch_size, :] = features
            self.queue_labels[ptr: ptr + batch_size] = labels
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    @torch.no_grad()
    def sample_target(self, features, labels):
        neighbor = []
        for i, label in enumerate(labels):
            pos = torch.where(self.queue_labels == label)[0]
            if len(pos) != 0:
                if self.num_positive > 0:
                    choice = torch.multinomial(torch.ones_like(pos).type(torch.FloatTensor), self.num_positive,
                                               replacement=True)
                    idx = pos[choice]
                    neighbor.append(self.queue[idx].mean(0))
                else:
                    neighbor.append(self.queue[pos].mean(0))
            else:
                neighbor.append(features[i])
        return torch.stack(neighbor, dim=0)

    def update_ema_model(self):
        ema_ratio = self.cfg.MOCO_MOMENTUM
        for param, param_ema in zip(self.network.parameters(), self.network_ema.parameters()):
            param_ema.data.mul_(ema_ratio).add_(param.data, alpha=1 - ema_ratio)

    def img_process(self, img_tensor, mask_tensor, fundusAug):
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)
        return img_tensor_new, img_tensor_ori

    def orthogonality_loss(self, z_tra, z_tia):
        # 计算两个特征向量的余弦相似度平方，并由 0 约束
        z_tra_norm = F.normalize(z_tra, dim=-1)
        z_tia_norm = F.normalize(z_tia, dim=-1)
        cosine = (z_tra_norm * z_tia_norm).sum(dim=-1)  # [B]
        loss_ortho = torch.mean(cosine ** 2)
        return loss_ortho

    def update(self, minibatch):
        self.global_step += 1
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        image_new, image_ori = self.img_process(image, mask, self.fundusAug)

        # 移动设备
        curr_device = image_new.device
        if next(self.projector.parameters()).device != curr_device:
            self.projector.to(curr_device)
            self.tra_projector.to(curr_device)
            self.tia_projector.to(curr_device)
            self.dino_classifier.to(curr_device)
            self.dino_projector.to(curr_device)
            self.predictor.to(curr_device)
            self.network_ema.to(curr_device)
            if self.queue.device != curr_device:
                self.queue = self.queue.to(curr_device)
                self.queue_labels = self.queue_labels.to(curr_device)
                self.queue_ptr = self.queue_ptr.to(curr_device)

        bs = self.cfg.BLOCK_SIZE
        patches = self.patchify(image_new, block=bs)
        mask_token, mask_inverse = self.random_masking(patches, self.cfg.MASK_RATIO)
        image_new_masked = self.unpatchify(patches * mask_inverse, block=bs)

        x1_splits = self._local_split(image_ori)
        x2_splits = self._local_split(image_new)

        # === 1. Full Model (Original View) ===
        # dino_ori 带有梯度 (得益于 nets.py 的修改)
        logits_ori, feats_ori, dino_ori = self.network(image_ori, force_resize=True)
        z_ori = self.projector(feats_ori)
        p_ori = self.predictor(z_ori)

        # --- DINOv3-FD Disentanglement ---
        # 1. 拆分特征
        tra_ori = self.tra_projector(dino_ori)  # Task Relevant
        tia_ori = self.tia_projector(dino_ori)  # Task Irrelevant

        # 2. FD Loss 计算
        # A. Classification Loss: TRA 必须能分类
        logits_tra_ori = self.dino_classifier(tra_ori)
        loss_dino_cls = F.cross_entropy(logits_tra_ori, label)

        # B. Orthogonality Loss: TRA 和 TIA 必须垂直
        loss_ortho = self.orthogonality_loss(tra_ori, tia_ori)

        # --- End FD ---

        # === 2. Full Model (Augmented View) ===
        logits_new, feats_new, dino_new = self.network(image_new, force_resize=True)
        z_new = self.projector(feats_new)
        p_new = self.predictor(z_new)

        tra_new = self.tra_projector(dino_new)

        # === 3. Masked View ===
        logits_masked, feats_masked, _ = self.network(image_new_masked, force_resize=True)
        z_masked = self.projector(feats_masked)
        p_masked = self.predictor(z_masked)

        # 归一化
        z_ori = nn.functional.normalize(z_ori, dim=-1)
        z_new = nn.functional.normalize(z_new, dim=-1)
        p_ori = nn.functional.normalize(p_ori, dim=-1)
        p_new = nn.functional.normalize(p_new, dim=-1)
        z_masked = nn.functional.normalize(z_masked, dim=-1)
        p_masked = nn.functional.normalize(p_masked, dim=-1)

        # === 4. DINO Distillation (Updated Interaction) ===
        # CNN (p_ori) 模仿 TRA 特征 (纯净病灶特征)
        # detach() 是为了不让蒸馏Loss干扰 FD 的拆分训练
        z_dino_tra_ori = self.dino_projector(tra_ori.detach())
        z_dino_tra_new = self.dino_projector(tra_new.detach())

        z_dino_tra_ori = nn.functional.normalize(z_dino_tra_ori, dim=-1)
        z_dino_tra_new = nn.functional.normalize(z_dino_tra_new, dim=-1)

        # === EMA & FastMoCo ===
        with torch.no_grad():
            _, feats_new_ema, _ = self.network_ema(image_new, force_resize=True)
            z_new_ema = self.projector(feats_new_ema)
            z_new_ema = nn.functional.normalize(z_new_ema, dim=-1)

        _, z1_split_feats, _ = self.network(x1_splits, force_resize=False)
        _, z2_split_feats, _ = self.network(x2_splits, force_resize=False)

        z1_split_list = list(z1_split_feats.split(image_ori.size(0), dim=0))
        z2_split_list = list(z2_split_feats.split(image_new.size(0), dim=0))
        z1_orthmix_ = torch.cat(
            list(map(lambda x: sum(x) / self.combs, list(combinations(z1_split_list, r=self.combs)))), dim=0)
        z2_orthmix_ = torch.cat(
            list(map(lambda x: sum(x) / self.combs, list(combinations(z2_split_list, r=self.combs)))), dim=0)
        z1_orthmix = nn.functional.normalize(self.projector(z1_orthmix_), dim=-1)
        z2_orthmix = nn.functional.normalize(self.projector(z2_orthmix_), dim=-1)
        p1_orthmix = nn.functional.normalize(self.predictor(z1_orthmix), dim=-1)
        p2_orthmix = nn.functional.normalize(self.predictor(z2_orthmix), dim=-1)

        # Loss Calculation
        z1_sup = self.sample_target(z_ori.detach(), label)
        z2_sup = self.sample_target(z_new_ema.detach(), label)

        feature_list = [feats_ori, feats_new, z_ori, z_new_ema, z1_sup, z2_sup, p_ori, p_new, z1_orthmix, z2_orthmix,
                        p1_orthmix, p2_orthmix]
        masked_list = [z_masked, None, p_masked, None]

        loss, loss_dict_iter = self.criterion([logits_new, logits_ori, logits_masked, logits_ori], feature_list,
                                              masked_list, label, domain, None)

        # Distillation Loss (Targeting TRA)
        loss_dino_1 = - (p_ori * z_dino_tra_ori).sum(dim=1).mean()
        loss_dino_2 = - (p_new * z_dino_tra_new).sum(dim=1).mean()
        loss_dino = 0.5 * (loss_dino_1 + loss_dino_2)

        # Total Loss Integration
        # DINO 分类权重建议设大一点，保证 LoRA 学会看病
        lambda_dino = self.cfg.GDRNET.LAMBDA_DINO
        lambda_fd_cls = 1.0
        lambda_fd_ortho = 0.1

        total_loss = loss + loss_dino * lambda_dino + \
                     loss_dino_cls * lambda_fd_cls + \
                     loss_ortho * lambda_fd_ortho

        # Logging
        loss_dict_iter['loss_dino_distill'] = loss_dino.item()
        loss_dict_iter['loss_fd_cls'] = loss_dino_cls.item()
        loss_dict_iter['loss_fd_ortho'] = loss_ortho.item()
        loss_dict_iter['loss'] = total_loss.item()

        total_loss.backward()
        self.optimizer.step()
        self.update_ema_model()
        self.dequeue_and_enqueue(z_new_ema.detach(), label)

        return loss_dict_iter

    # ... (predict, save_model 等保持不变) ...
    def predict(self, x):
        out = self.network(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.network_ema.state_dict(), os.path.join(log_path, 'best_model_ema.pth'))

    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        state_dict = torch.load(net_path)
        self.network.load_state_dict(state_dict)

class GREEN(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GREEN, self).__init__(num_classes, cfg)
        self.network = models.get_net(cfg)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr = cfg.LEARNING_RATE,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        output = self.network(image)
        loss = F.cross_entropy(output, label)

        loss.backward()
        self.optimizer.step()
        return {'loss':loss}

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

        return {'loss':loss}

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
        self.ema_per_domain = [misc.MovingAverage(cfg.FISHR.EMA, oneminusema_correction=True) for _ in range(self.num_groups)]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            list(self.network.parameters()) + list(self.classifier.parameters()),
            lr = self.cfg.LEARNING_RATE,
            momentum = self.cfg.MOMENTUM,
            weight_decay = self.cfg.WEIGHT_DECAY,
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
        dict_grads = OrderedDict([(name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)) for name, weights in self.classifier.named_parameters()])
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
        grads_var = OrderedDict([(name, torch.stack([grads_var_per_domain[domain_id][name] for domain_id in range(self.num_groups)], dim=0).mean(dim=0)) for name in grads_var_per_domain[0].keys()])
        penalty = 0
        for domain_id in range(self.num_groups):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_groups

    def l2_between_dicts(self, dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (torch.cat(tuple([t.view(-1) for t in dict_1_values])) - torch.cat(tuple([t.view(-1) for t in dict_2_values]))).pow(2).mean()

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
            self.swad_algorithm.update_parameters(self.algorithm, step = self.epoch)
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
                    swad_val_auc, swad_val_loss = algorithm_validate(self.swad_algorithm, val_loader, writer, self.epoch, 'val')
                    swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.epoch, 'test')

                    if hasattr(self.swad, "dead_valley") and self.swad.dead_valley:
                        logging.info("SWAD valley is dead -> not stop !")
                    
                    self.swad_algorithm = AveragedModel(self.algorithm)  # reset
            
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
                
        else:
            self.swad_algorithm = self.swad.get_final_model()
            logging.warning("Evaluate SWAD ...")
            swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH , 'test')
            logging.info('(last) swad test auc: {}  loss: {}'.format(swad_auc,swad_loss))
            
        return swad_val_auc, swad_auc    
        
    def save_model(self, log_path):
        self.algorithm.save_model(log_path)
    
    def renew_model(self, log_path):
        self.algorithm.renew_model(log_path)
    
    def predict(self, x):
        return self.swad_algorithm.predict(x)