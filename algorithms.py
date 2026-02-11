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
        val_metrics = algorithm_validate(self, val_loader, writer, self.epoch, 'val')

        if test_loader is not None:
            test_metrics = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
        else:
            test_metrics = {'auc': 0.0, 'acc': 0.0, 'f1': 0.0, 'loss': 0.0}

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


class GDRNet(ERM):
    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)

        # 1. 主干网络 (FPTPlusNet: SideResNet + Frozen DINOv3)
        self.network = models.get_net(cfg)

        # 2. EMA Teacher (用于 Feature Level 对比学习，构建稳定的 Target)
        self.network_ema = deepcopy(self.network)
        for param in self.network_ema.parameters():
            param.requires_grad = False

        dim_in = 2048  # ResNet50 output
        feat_dim = 512  # Projector output dim
        dino_dim = 768  # DINOv3 hidden dim

        # [交互组件 1: CNN Projector & Predictor]
        # 用于 Student (CNN) 的特征投影
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim, bias=False), nn.BatchNorm1d(feat_dim, affine=False)
        )
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        # [交互组件 2: DINOv3-FD (Feature Disentanglement)]
        print(f">> [GDRNet] Initializing DINOv3-FD Full Suite...")
        # TRA (Task-Relevant) 投影头
        self.tra_projector = nn.Sequential(
            nn.Linear(dino_dim, dino_dim // 2), nn.LayerNorm(dino_dim // 2), nn.GELU(),
            nn.Linear(dino_dim // 2, dino_dim)
        )
        # TIA (Task-Irrelevant) 投影头
        self.tia_projector = nn.Sequential(
            nn.Linear(dino_dim, dino_dim // 2), nn.LayerNorm(dino_dim // 2), nn.GELU(),
            nn.Linear(dino_dim // 2, dino_dim)
        )

        # [新增] 将 DINO TRA 特征映射到 Contrastive Space (用于与 CNN 对齐)
        self.dino_contrast_projector = nn.Sequential(
            nn.Linear(dino_dim, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(),
            nn.Linear(dim_in, feat_dim)
        )

        # DINO Classifier (用于生成 Logit 软标签)
        self.dino_classifier = nn.Linear(dino_dim, num_classes)

        # [新增] DINO Predictor (用于 Saliency Masking 后特征对齐的辅助)
        self.dino_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        # 优化器
        self.optimizer = torch.optim.Adam([
            {"params": self.network.parameters(), 'fix_lr': False},
            {"params": self.projector.parameters(), 'fix_lr': False},
            {"params": self.predictor.parameters(), 'fix_lr': True},
            {"params": self.tra_projector.parameters(), 'fix_lr': False},
            {"params": self.tia_projector.parameters(), 'fix_lr': False},
            {"params": self.dino_classifier.parameters(), 'fix_lr': False},
            {"params": self.dino_contrast_projector.parameters(), 'fix_lr': False},
            {"params": self.dino_predictor.parameters(), 'fix_lr': False},
        ], lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

        self.fundusAug = get_post_FundusAug(cfg)
        self.global_step = 0

        # MoCo Queue (Feature Level Contrast)
        self.K = cfg.MOCO_QUEUE_K
        self.register_buffer("queue", torch.randn(self.K, feat_dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        print(">> [Optimization] H100 Detected. Switching to BFloat16 Training.")

        # ============================================================
        # [方案一] 注册 Hook 以获取 SideResNet layer4 的空间特征图 (用于 Attention Transfer)
        # ============================================================
        self.student_feats = None

        def hook_fn(module, input, output):
            # output shape: (B, 2048, 7, 7) assuming 224x224 input
            self.student_feats = output

        # 确保我们要 hook 的层存在 (FPTPlusNet -> model -> layer4)
        if hasattr(self.network, 'model') and hasattr(self.network.model, 'layer4'):
            self.network.model.layer4.register_forward_hook(hook_fn)
            print(">> [GDRNet] Hook registered on SideResNet layer4 for Attention Transfer.")
        else:
            print(">> [Warning] Could not find layer4 in network. Attention Transfer might fail.")

    def at_loss(self, student_feats, teacher_attn):
        """
        Attention Transfer Loss (Option 1)
        """
        # shape: (B, H_s, W_s)
        stu_map = torch.mean(student_feats, dim=1)

        B, N = teacher_attn.shape
        side = int(N ** 0.5)
        # shape: (B, H_t, W_t)
        tea_map = teacher_attn.reshape(B, side, side)

        stu_map = F.interpolate(stu_map.unsqueeze(1), size=(side, side), mode='bilinear', align_corners=False).squeeze(
            1)

        stu_vec = F.normalize(stu_map.flatten(1), dim=1)
        tea_vec = F.normalize(tea_map.flatten(1), dim=1)

        return F.mse_loss(stu_vec, tea_vec)

    def compute_saliency_score(self, attn_map):
        """
        attn_map: (B, N_patches) - DINOv3 的 CLS Token Attention
        逻辑: Noise Injection -> MinMax Norm -> Scale to 0.5-1.0
        """
        B, N = attn_map.shape

        # 1. 噪声注入 (Noise Injection)
        noise = torch.randint(-5000, 5000, attn_map.shape, device=attn_map.device).float()
        attn_map = attn_map + noise * 0.0001

        # 2. 归一化 (Min-Max Normalization)
        min_v, _ = torch.min(attn_map, dim=1, keepdim=True)
        max_v, _ = torch.max(attn_map, dim=1, keepdim=True)
        scores = (attn_map - min_v) / (max_v - min_v + 1e-6)

        # 3. 缩放 (Scaling)
        scores = scores / 2.0 + 0.5

        return scores.to(dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # [SPMix 核心复现 2] 基于 Saliency 的 Mask 生成
    # ------------------------------------------------------------------
    def get_saliency_guided_mask(self, scores, mask_ratio=0.5, mask_mode='hard'):
        """
        scores: (B, N)
        mask_mode:
            'hard': Mask 掉分数最高的区域 (用于 Logit Distill，迫使 Student 学 Context)
            'easy': Mask 掉分数最低的区域
        """
        B, N = scores.shape
        num_mask = int(N * mask_ratio)

        if mask_mode == 'hard':
            # 找出分数最高的索引 (显著区域)
            _, idx = torch.topk(scores, num_mask, dim=1, largest=True)
        else:
            # 找出分数最低的索引 (背景区域)
            _, idx = torch.topk(scores, num_mask, dim=1, largest=False)

        mask = torch.ones((B, N), device=scores.device)
        # 将选中的区域置为 0 (Masked)
        mask.scatter_(1, idx, 0)

        return mask.unsqueeze(-1), (1 - mask.unsqueeze(-1))  # (B, N, 1)

    # 辅助函数：Patchify
    def patchify(self, imgs, block=14):
        p = block
        n, c, h, w = imgs.shape
        # Ensure dimensions match block size
        if h % p != 0 or w % p != 0:
            h_new = (h // p) * p
            w_new = (w // p) * p
            imgs = F.interpolate(imgs, size=(h_new, w_new), mode='bilinear', align_corners=False)

        x = imgs.reshape(n, c, h // p, p, w // p, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(n, (h // p) * (w // p), p * p * c)
        return x

    # 辅助函数：Unpatchify
    def unpatchify(self, x, block=14):
        p = block
        n, l, d = x.shape
        h = w = int(l ** 0.5)
        x = x.reshape(n, h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(n, 3, h * p, h * p)
        return imgs

    # 辅助函数：MoCo Queue 更新
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            rem = self.K - ptr
            self.queue[ptr:self.K] = keys[:rem]
            self.queue[0:batch_size - rem] = keys[rem:]
        else:
            self.queue[ptr:ptr + batch_size] = keys
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def update_ema_model(self):
        m = 0.999  # Momentum
        for param_q, param_k in zip(self.network.parameters(), self.network_ema.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.data)

    def img_process(self, img_tensor, mask_tensor, fundusAug):
        img_tensor_new, _ = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)
        return img_tensor_new, img_tensor_ori

    def orthogonality_loss(self, z_tra, z_tia):
        z_tra_norm = F.normalize(z_tra, dim=-1)
        z_tia_norm = F.normalize(z_tia, dim=-1)
        cosine = (z_tra_norm * z_tia_norm).sum(dim=-1)
        return torch.mean(cosine ** 2)

    def update(self, minibatch, step=None, accum_iter=1):
        if step is None: step = self.global_step
        self.global_step += 1
        image, mask, label, domain = minibatch

        if step % accum_iter == 0: self.optimizer.zero_grad()

        # 1. 基础增强
        image_aug, image_ori = self.img_process(image, mask, self.fundusAug)
        curr_device = image.device

        # 确保 Projector 等在正确设备
        if next(self.projector.parameters()).device != curr_device:
            self.projector.to(curr_device)
            self.predictor.to(curr_device)
            self.tra_projector.to(curr_device)
            self.tia_projector.to(curr_device)
            self.dino_contrast_projector.to(curr_device)
            self.dino_classifier.to(curr_device)
            self.dino_predictor.to(curr_device)
            self.network_ema.to(curr_device)
            self.queue = self.queue.to(curr_device)
            self.queue_ptr = self.queue_ptr.to(curr_device)

        with autocast(dtype=torch.bfloat16):
            # ============================================================
            # Phase 1: Teacher (DINOv3) 提取全局信息 & Saliency Generation
            # ============================================================
            with torch.no_grad():
                # [Fix] Handle DDP wrapper
                if isinstance(self.network,
                              (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
                    raw_network = self.network.module
                else:
                    raw_network = self.network

                # 调用 frozen_encoder 获取 Attention
                dino_outputs = raw_network.frozen_encoder(image_ori, output_attentions=True)

                # 获取 CLS Token 特征
                if hasattr(dino_outputs, 'last_hidden_state'):
                    dino_feat_full = dino_outputs.last_hidden_state[:, 0, :]
                else:
                    dino_feat_full = dino_outputs[0][:, 0, :]

                # 获取 Attention Map (最后一层)
                attentions = dino_outputs.attentions[-1]

                # 取 CLS token 对后续 token 的 Attention
                raw_attn = attentions[:, :, 0, 1:]  # (B, H_heads, N_tokens_all)

                # 自动剔除 Register Tokens
                n_tokens_all = raw_attn.shape[-1]
                side = int(n_tokens_all ** 0.5)
                n_patches = side ** 2

                if n_tokens_all > n_patches:
                    cls_attn = raw_attn[:, :, -n_patches:]
                else:
                    cls_attn = raw_attn

                # [方案一关键] 获取原始的 Saliency Map 用于 Attention Transfer
                # Mean over heads
                saliency_map_raw = cls_attn.mean(dim=1)  # (B, N_patches)

                # 计算用于 Masking 的 Score (加了噪声的)
                saliency_scores = self.compute_saliency_score(saliency_map_raw)

            # ============================================================
            # Phase 2: Saliency-Guided Masking
            # ============================================================
            # Logit Level Distill 预备: Mask 掉最显著区域
            mask_token, mask_inverse = self.get_saliency_guided_mask(
                saliency_scores,
                mask_ratio=self.cfg.MASK_RATIO,
                mask_mode='hard'
            )

            p_size = 14  # DINOv3 默认 Patch Size

            # 插值调整 Saliency Score 维度 (若 image_aug 与 image_ori 分辨率不同)
            H_img, W_img = image_aug.shape[2], image_aug.shape[3]
            grid_h, grid_w = H_img // p_size, W_img // p_size

            # 注意：如果 image_aug 分辨率变了，我们需要对 saliency_scores 进行插值
            # 同时也需要对 saliency_map_raw 进行插值以用于 AT Loss
            if saliency_scores.shape[1] != grid_h * grid_w:
                N_dino = saliency_scores.shape[1]
                side = int(N_dino ** 0.5)

                # 1. Resize Scores for Masking
                scores_reshaped = saliency_scores.reshape(saliency_scores.shape[0], 1, side, side)
                scores_interpolated = F.interpolate(scores_reshaped, size=(grid_h, grid_w), mode='bilinear')
                saliency_scores_resized = scores_interpolated.flatten(1)

                # 2. Resize Raw Map for AT Loss
                raw_map_reshaped = saliency_map_raw.reshape(saliency_map_raw.shape[0], 1, side, side)
                raw_map_interpolated = F.interpolate(raw_map_reshaped, size=(grid_h, grid_w), mode='bilinear')
                saliency_map_for_at = raw_map_interpolated.flatten(1)

                # 重新生成 Mask (尺寸对齐后)
                mask_token, mask_inverse = self.get_saliency_guided_mask(
                    saliency_scores_resized,
                    mask_ratio=self.cfg.MASK_RATIO,
                    mask_mode='hard'
                )
            else:
                saliency_map_for_at = saliency_map_raw

            # 应用 Mask
            x_patches = self.patchify(image_aug, block=p_size)
            image_masked = self.unpatchify(x_patches * mask_token, block=p_size)

            # ============================================================
            # Phase 3: Forward Student (CNN/SideNet) & Teacher
            # ============================================================
            # 1. Student 看 Masked Image (用于 Logit Distillation)
            logits_student_masked, feats_student_masked, _ = self.network(image_masked)

            # 2. Student 看 Full Augmented Image (用于 Feature Contrast & Attention Transfer)
            # [方案一 Hook] Hook 会在这里被触发，self.student_feats 会被赋值
            logits_student_full, feats_student_full, _ = self.network(image_aug)

            # 3. Teacher 生成 TRA 特征和 Soft Labels
            # [Fix] 必须允许梯度，因为我们需要通过 loss_fd_cls 训练 tra_projector 和 dino_classifier
            tra_full = self.tra_projector(dino_feat_full)
            logits_teacher = self.dino_classifier(tra_full)

            # ============================================================
            # Phase 4: Loss Calculation
            # ============================================================

            # 1. Classification Loss (GT)
            loss_cls = F.cross_entropy(logits_student_full, label)

            # 2. [方案三] Logit Level Distillation (Reliability-Weighted)
            T = 2.0

            # 计算 Teacher 熵 (作为可靠性权重)
            # 使用 Detach 避免通过权重反向传播影响 Teacher 的置信度
            with torch.no_grad():
                probs_teacher = F.softmax(logits_teacher, dim=1)
                entropy = -torch.sum(probs_teacher * torch.log(probs_teacher + 1e-6), dim=1)
                # 熵越小(越自信) -> 权重越大; 熵越大(瞎猜) -> 权重越小
                weight = torch.exp(-entropy)

            # 计算 KL Div (不 reduce，保留 batch 维度)
            # [Standard KD] Target (Teacher) 应该 detach，不让 Student 训练 Teacher
            loss_logit_distill_raw = F.kl_div(
                F.log_softmax(logits_student_masked / T, dim=1),
                F.softmax(logits_teacher.detach() / T, dim=1),
                reduction='none'
            ).sum(dim=1) * (T * T)

            # 应用权重并平均
            loss_logit_distill = (loss_logit_distill_raw * weight).mean()

            # 3. Feature Level Contrast (MoCo Style) - 正面拉近
            z_student = self.predictor(self.projector(feats_student_full))
            z_student = F.normalize(z_student, dim=1)

            with torch.no_grad():
                z_teacher_tra = self.dino_contrast_projector(tra_full)
                z_teacher_tra = F.normalize(z_teacher_tra, dim=1)

            l_pos = torch.einsum('nc,nc->n', [z_student, z_teacher_tra]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [z_student, self.queue.clone().detach().t()])

            logits_con = torch.cat([l_pos, l_neg], dim=1)
            logits_con /= 0.07  # temperature

            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long, device=curr_device)
            loss_contrast = F.cross_entropy(logits_con, labels_con)

            # Update Queue
            self._dequeue_and_enqueue(z_teacher_tra)

            # 4. Feature Disentanglement Loss (Teacher 内部约束)
            tia_full = self.tia_projector(dino_feat_full)
            loss_ortho = self.orthogonality_loss(tra_full, tia_full)

            # 5. [方案二] Negative Distillation (负面蒸馏：推开 TIA)
            # 我们希望 Student 的特征 (z_student) 与 Teacher 的噪声特征 (tia) 正交
            with torch.no_grad():
                # 关键：使用同一个 contrast_projector 将 TIA 投影到和 Student 相同的空间
                z_teacher_tia = self.dino_contrast_projector(tia_full)
                z_teacher_tia = F.normalize(z_teacher_tia, dim=1)

            # 计算余弦相似度
            similarity_noise = torch.sum(z_student * z_teacher_tia, dim=1)
            # 最小化相似度的平方 (逼近 0)
            loss_neg_distill = torch.mean(similarity_noise ** 2)

            # 6. TRA Classification Loss (用于训练 Teacher Projector)
            loss_fd_cls = F.cross_entropy(logits_teacher, label)

            # 7. [方案一] Attention Transfer Loss
            # 确保 Hook 捕获到了特征
            if self.student_feats is not None:
                loss_at = self.at_loss(self.student_feats, saliency_map_for_at)
            else:
                loss_at = torch.tensor(0.0, device=curr_device)

            # 权重设置 (建议后续移入 Config)
            lambda_distill = 1.0
            lambda_contrast = 0.5
            lambda_ortho = 0.1
            lambda_fd_cls = 0.5
            lambda_at = 1000.0  # Attention Transfer 权重
            lambda_neg = 0.5  # Negative Distillation 权重

            total_loss = loss_cls + \
                         lambda_distill * loss_logit_distill + \
                         lambda_contrast * loss_contrast + \
                         lambda_ortho * loss_ortho + \
                         lambda_fd_cls * loss_fd_cls + \
                         lambda_at * loss_at + \
                         lambda_neg * loss_neg_distill

            total_loss = total_loss / accum_iter

        # Logging & Backward
        loss_dict_iter = {'loss': total_loss.item() * accum_iter,
                          'cls': loss_cls.item(),
                          'distill': loss_logit_distill.item(),
                          'contrast': loss_contrast.item(),
                          'ortho': loss_ortho.item(),
                          'at': loss_at.item(),
                          'neg': loss_neg_distill.item()}

        total_loss.backward()

        if step % accum_iter == 0:
            self.optimizer.step()
            self.update_ema_model()
            # 清理 Hook 内容，防止显存泄漏
            self.student_feats = None

        return loss_dict_iter

    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.network_ema.state_dict(), os.path.join(log_path, 'best_model_ema.pth'))

    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        state_dict = torch.load(net_path)
        self.network.load_state_dict(state_dict)

    def predict(self, x):
        out = self.network(x)
        while isinstance(out, (tuple, list)):
            out = out[0]
        return out


class GREEN(Algorithm):
    # ... (保持不变) ...
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


# ... (其余 Algorithm 子类保持不变, 复制即可) ...
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