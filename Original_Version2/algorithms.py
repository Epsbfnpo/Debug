import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, collections, logging
from collections import OrderedDict
import utils.misc as misc
from utils.validate import algorithm_validate
import modeling.model_manager as models
from modeling.losses import DahLoss, GDRNetLoss_Integrated, SupConLoss
from modeling.nets import LossValley, AveragedModel, DualTowerGDRNet
from dataset.data_manager import get_post_FundusAug
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from itertools import combinations
import torch.distributed as dist
import copy
import contextlib
import math
import torchvision.transforms.v2 as v2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ALGORITHMS = ['ERM', 'GDRNet', 'GREEN', 'CABNet', 'MixupNet', 'MixStyleNet', 'Fishr', 'DRGen', 'CASS_GDRNet']


def compute_attention_transfer_loss(map_cnn, map_vit):
    at_cnn = map_cnn.pow(2).mean(dim=1, keepdim=True)
    at_vit = map_vit.pow(2).mean(dim=1, keepdim=True)
    at_cnn_resized = F.interpolate(at_cnn, size=at_vit.shape[2:], mode='bilinear', align_corners=False)
    at_cnn_flat = at_cnn_resized.view(at_cnn_resized.size(0), -1)
    at_vit_flat = at_vit.view(at_vit.size(0), -1)
    cos_sim = F.cosine_similarity(at_cnn_flat, at_vit_flat, dim=1)
    return (1.0 - cos_sim).mean()


def compute_entropy(probs):
    """计算预测分布的信息熵。熵越高越不确定，熵越低越自信。"""
    return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()


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
        raise NotImplementedError

    def save_model(self, log_path, **kwargs):
        raise NotImplementedError

    def renew_model(self, log_path, **kwargs):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    def __init__(self, num_classes, cfg):
        super(ERM, self).__init__(num_classes, cfg)
        self.network = models.get_net(cfg)
        self.classifier = models.get_classifier(self.network.out_features(), cfg)
        self.optimizer = torch.optim.Adam([{"params": self.network.parameters()}, {"params": self.classifier.parameters()}], lr=cfg.LEARNING_RATE, weight_decay=0.0001)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()
        features = self.network(image)
        output = self.classifier(features)
        loss = F.cross_entropy(output, label)
        loss.backward()
        self.optimizer.step()
        return {'loss':loss}
    
    def validate(self, val_loader, test_loader, writer):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
        return val_auc, test_auc

    def save_model(self, log_path, **kwargs):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))

    def renew_model(self, log_path, **kwargs):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        return self.classifier(self.network(x))


class GDRNet(ERM):
    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)
        dim_in = self.network.out_features()
        feat_dim = 512
        self.projector = nn.Sequential(nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in, affine=False))
        self.predictor = nn.Sequential(nn.Linear(dim_in, feat_dim, bias=False), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, dim_in))
        self.K = 1024
        self.register_buffer("queue", torch.randn(self.K, dim_in))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(self.K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.split_num = 2
        self.combs = 3
        self.fundusAug = get_post_FundusAug(cfg)
        # 对齐 GDRNetLoss_Integrated 最新签名，避免运行 GDRNet 基线时参数不匹配
        self.criterion = GDRNetLoss_Integrated(training_domains=cfg.DATASET.SOURCE_DOMAINS, beta=cfg.GDRNET.BETA)
        self.optimizer = torch.optim.Adam([{"params": self.network.parameters()}, {"params": self.classifier.parameters()}, {"params": self.projector.parameters()}, {"params": self.predictor.parameters()}], lr=cfg.LEARNING_RATE, weight_decay=0.0001)

    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        if self.K % batch_size != 0:
            pass
        replace_idx = torch.arange(ptr, ptr + batch_size).cuda() % self.K
        self.queue[replace_idx, :] = features
        self.queue_labels[replace_idx] = labels
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def sample_target(self, features, labels):
        neighbor = []
        for i, label in enumerate(labels):
            pos = torch.where(self.queue_labels == label)[0]
            if len(pos) != 0:
                neighbor.append(self.queue[pos].mean(0))
            else:
                neighbor.append(features[i])
        return torch.stack(neighbor, dim=0)

    def _local_split(self, x):
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)
        return torch.cat(xs, dim=0)

    def img_process(self, img_tensor, mask_tensor, fundusAug):
        img_new, mask_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_new = img_new * mask_new
        img_new = fundusAug['post_aug2'](img_new)
        img_ori = fundusAug['post_aug2'](img_tensor)
        return img_new, img_ori

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()
        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        features_ori = self.network(image_ori)
        features_new = self.network(image_new)
        output_logits = self.classifier(features_new)
        z1 = self.projector(features_ori)
        z2 = self.projector(features_new)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        z1_sup = self.sample_target(z1.detach(), label)
        z2_sup = self.sample_target(z2.detach(), label)
        x1_split = self._local_split(image_ori)
        x2_split = self._local_split(image_new)
        z1_pre = self.network(x1_split)
        z2_pre = self.network(x2_split)
        chunk_size = z1_pre.size(0) // (self.split_num ** 2)
        z1_splits = list(z1_pre.split(chunk_size, dim=0))
        z2_splits = list(z2_pre.split(chunk_size, dim=0))
        z1_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z1_splits, r=self.combs)))), dim=0)
        z2_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z2_splits, r=self.combs)))), dim=0)
        z1_orthmix_proj = self.projector(z1_orthmix_)
        z2_orthmix_proj = self.projector(z2_orthmix_)
        p1_orthmix_ = self.predictor(z1_orthmix_proj)
        p2_orthmix_ = self.predictor(z2_orthmix_proj)
        num_mixs = len(list(combinations(range(self.split_num ** 2), self.combs)))
        p1_orthmix_list = list(p1_orthmix_.split(image.size(0), dim=0))
        p2_orthmix_list = list(p2_orthmix_.split(image.size(0), dim=0))
        p1_orthmix_list = [F.normalize(p, dim=-1) for p in p1_orthmix_list]
        p2_orthmix_list = [F.normalize(p, dim=-1) for p in p2_orthmix_list]
        self.dequeue_and_enqueue(z2.detach(), label)
        contrastive_features = [z1, z2, z1_sup, z2_sup, p1, p2, p1_orthmix_list, p2_orthmix_list]
        loss, loss_dict_iter = self.criterion(output_logits, contrastive_features, label, domain)
        loss.backward()
        self.optimizer.step()
        return loss_dict_iter

    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)

class GREEN(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GREEN, self).__init__(num_classes, cfg)
        self.network = models.get_net(cfg)
        self.optimizer = torch.optim.Adam([{"params": self.network.parameters()}, {"params": self.classifier.parameters()}], lr=cfg.LEARNING_RATE, weight_decay=0.0001)
    
    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()
        output = self.network(image)
        loss = F.cross_entropy(output, label)
        loss.backward()
        self.optimizer.step()
        return {'loss':loss}
    
    def validate(self, val_loader, test_loader, writer):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
        return val_auc, test_auc
    
    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        self.network.load_state_dict(torch.load(net_path))
    
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
        self.optimizer = torch.optim.Adam([{"params": self.network.parameters()}, {"params": self.classifier.parameters()}], lr=cfg.LEARNING_RATE, weight_decay=0.0001)
        
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

class CASS_GDRNet(Algorithm):
    def __init__(self, num_classes, cfg):
        super(CASS_GDRNet, self).__init__(num_classes, cfg)
        self.network = DualTowerGDRNet(cfg)
        self.momentum_network = copy.deepcopy(self.network)
        for param in self.momentum_network.parameters():
            param.requires_grad = False
        self.m = 0.999

        self.base_lr_backbone = 5e-5
        self.base_lr_head = 5e-4
        self.warmup_epochs = 5

        head_modules = [
            self.network.projector_cnn, self.network.projector_vit,
            self.network.predictor_cnn, self.network.predictor_vit,
            self.network.classifier_cnn, self.network.classifier_vit,
            self.network.dual_stream_neck
        ]
        head_params = []
        for module in head_modules:
            if module is not None:
                head_params.extend([p for p in module.parameters() if p.requires_grad])
        self.head_params = head_params
        head_param_ids = {id(p) for p in self.head_params}

        self.backbone_params = [
            p for p in self.network.parameters()
            if p.requires_grad and id(p) not in head_param_ids
        ]

        self.optimizer = AdamW(
            [
                {'params': self.backbone_params, 'lr': self.base_lr_backbone},
                {'params': self.head_params, 'lr': self.base_lr_head}
            ],
            weight_decay=5e-2
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.cfg.EPOCHS - self.warmup_epochs),
            eta_min=1e-6
        )

        self.max_epochs = cfg.EPOCHS
        self.K = 1024
        proj_dim = 1024
        self.num_positive = getattr(cfg, 'POSITIVE', 4)

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        batch_size_per_gpu = getattr(cfg, 'BATCH_SIZE', 16)
        assert self.K % (batch_size_per_gpu * world_size) == 0, (
            f"队列容量 K ({self.K}) 必须被全局 Batch Size ({batch_size_per_gpu * world_size}) 整除！"
        )

        self.register_buffer("queue_cnn", torch.randn(self.K, proj_dim))
        self.queue_cnn = nn.functional.normalize(self.queue_cnn, dim=-1)
        self.register_buffer("queue_vit", torch.randn(self.K, proj_dim))
        self.queue_vit = nn.functional.normalize(self.queue_vit, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(self.K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.prototype_momentum = getattr(cfg, 'PROTOTYPE_MOMENTUM', 0.9)
        num_classes = cfg.DATASET.NUM_CLASSES
        self.register_buffer("class_proto_cnn", F.normalize(torch.randn(num_classes, proj_dim), dim=-1))
        self.register_buffer("class_proto_vit", F.normalize(torch.randn(num_classes, proj_dim), dim=-1))
        self.register_buffer("class_proto_initialized", torch.zeros(num_classes, dtype=torch.bool))

        self.split_num = 2
        self.combs = 3
        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = GDRNetLoss_Integrated(training_domains=cfg.DATASET.SOURCE_DOMAINS, beta=cfg.GDRNET.BETA)
        self.eval_branch = 'cnn'

        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=not self.use_bf16)

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.cnn_train_transforms = v2.Compose([
            v2.RandomResizedCrop(1024, scale=(0.6, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(45),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.3),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.vit_train_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.weak_transforms = v2.Compose([
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.weak_transforms_cnn = v2.Compose([
            v2.Resize((1024, 1024), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _apply_batch_transform(self, images, transform):
        return torch.stack([transform(img) for img in images], dim=0)

    def _compute_grad_norms(self):
        grad_norm_backbone = 0.0
        grad_norm_head = 0.0

        for p in self.backbone_params:
            if p.grad is not None:
                grad_norm_backbone += p.grad.data.norm(2).item() ** 2

        for p in self.head_params:
            if p.grad is not None:
                grad_norm_head += p.grad.data.norm(2).item() ** 2

        return grad_norm_backbone ** 0.5, grad_norm_head ** 0.5

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        if not dist.is_initialized():
            return tensor
        tensor = tensor.contiguous()
        tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)

    @torch.no_grad()
    def dequeue_and_enqueue(self, feat_cnn, feat_vit, labels):
        feat_cnn = self.concat_all_gather(feat_cnn.float())
        feat_vit = self.concat_all_gather(feat_vit.float())
        labels = self.concat_all_gather(labels)

        batch_size = feat_cnn.shape[0]
        ptr = int(self.queue_ptr)
        replace_idx = torch.arange(ptr, ptr + batch_size).to(feat_cnn.device) % self.K

        self.queue_cnn[replace_idx, :] = feat_cnn
        self.queue_vit[replace_idx, :] = feat_vit
        self.queue_labels[replace_idx] = labels
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    @torch.no_grad()
    def sample_queue_mean(self, current_features, labels, target_queue, class_prototypes):
        """
        Grade-aware 专属靶子生成器：只负责提取同类别的历史共识。
        包含冷启动安全机制：如果队列中没有同类，平滑退化为当前特征。
        """
        targets = []
        for i, label in enumerate(labels):
            class_idx = int(label.item())
            if class_idx < class_prototypes.shape[0] and self.class_proto_initialized[class_idx]:
                targets.append(class_prototypes[class_idx])
                continue

            pos = torch.where(self.queue_labels == label)[0]

            if len(pos) != 0:
                if getattr(self, 'num_positive', 4) > 0:
                    weights = torch.ones_like(pos).float()
                    choice = torch.multinomial(weights, self.num_positive, replacement=True)
                    queue_mean = target_queue[pos[choice]].mean(0)
                else:
                    queue_mean = target_queue[pos].mean(0)
                targets.append(queue_mean)
            else:
                targets.append(current_features[i])

        return F.normalize(torch.stack(targets, dim=0), dim=-1, eps=1e-6)

    @torch.no_grad()
    def _update_class_prototypes(self, feat_cnn, feat_vit, labels):
        feat_cnn = self.concat_all_gather(feat_cnn.float())
        feat_vit = self.concat_all_gather(feat_vit.float())
        labels = self.concat_all_gather(labels)

        unique_labels = torch.unique(labels)
        for cls in unique_labels:
            cls_idx = int(cls.item())
            mask = labels == cls
            if mask.sum() == 0:
                continue

            batch_center_cnn = F.normalize(feat_cnn[mask].mean(dim=0), dim=0, eps=1e-6)
            batch_center_vit = F.normalize(feat_vit[mask].mean(dim=0), dim=0, eps=1e-6)

            if not self.class_proto_initialized[cls_idx]:
                self.class_proto_cnn[cls_idx] = batch_center_cnn
                self.class_proto_vit[cls_idx] = batch_center_vit
                self.class_proto_initialized[cls_idx] = True
            else:
                mom = self.prototype_momentum
                self.class_proto_cnn[cls_idx] = F.normalize(
                    mom * self.class_proto_cnn[cls_idx] + (1.0 - mom) * batch_center_cnn,
                    dim=0,
                    eps=1e-6,
                )
                self.class_proto_vit[cls_idx] = F.normalize(
                    mom * self.class_proto_vit[cls_idx] + (1.0 - mom) * batch_center_vit,
                    dim=0,
                    eps=1e-6,
                )

    def _to_pixel_space(self, x):
        if x.min() < 0.0 or x.max() > 1.0:
            return (x * self.imagenet_std) + self.imagenet_mean
        return x

    def _cnn_normalize(self, x):
        return (x - self.imagenet_mean) / self.imagenet_std

    def _vit_normalize(self, x):
        return (x - self.imagenet_mean) / self.imagenet_std

    def _ensure_cnn_normalized(self, x):
        if x.min() < 0.0 or x.max() > 1.0:
            return x
        return self._cnn_normalize(x)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        network_inner = self.network.module if hasattr(self.network, 'module') else self.network
        momentum_inner = self.momentum_network.module if hasattr(self.momentum_network, 'module') else self.momentum_network

        image_pixel = self._to_pixel_space(image.clone()).clamp(0.0, 1.0)
        bg_color = torch.tensor([0.5074, 0.2816, 0.1456], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        mask_float = (mask > 0).to(image.dtype)
        if mask_float.shape[-2:] != image_pixel.shape[-2:]:
            mask_float = F.interpolate(mask_float, size=image_pixel.shape[-2:], mode='nearest')

        img_base_pixel = image_pixel * mask_float + bg_color * (1.0 - mask_float)
        img_weak_cnn = self.weak_transforms_cnn(img_base_pixel.clone()).contiguous()
        img_weak_vit = self.weak_transforms(img_base_pixel.clone()).contiguous()
        img_strong_cnn = self.cnn_train_transforms(img_base_pixel.clone()).contiguous()
        img_strong_vit = self.vit_train_transforms(img_base_pixel.clone()).contiguous()

        autocast_ctx = contextlib.nullcontext
        if torch.cuda.is_available():
            autocast_ctx = lambda: torch.amp.autocast('cuda', dtype=self.amp_dtype)

        with autocast_ctx():
            res_combined = self.network(
                x_cnn=img_strong_cnn,
                x_vit=img_strong_vit,
                return_train_features=True
            )

        res_clean_fp32 = {
            'proj_cnn': res_combined['proj_cnn'].float(),
            'proj_vit': res_combined['proj_vit'].float(),
            'pred_cnn': res_combined['pred_cnn'].float(),
            'pred_vit': res_combined['pred_vit'].float(),
            'logits_cnn': res_combined['logits_cnn'].float(),
            'logits_vit': res_combined['logits_vit'].float(),
            'tia_cls': res_combined['tia_cls'].float(),
            'spatial_tokens': res_combined['spatial_tokens'].float(),
            'feat_vit': res_combined['feat_vit'].float(),
        }

        with torch.no_grad():
            momentum_prev_mode = momentum_inner.training
            momentum_inner.eval()
            with autocast_ctx():
                res_momentum = momentum_inner(
                    x_cnn=img_weak_cnn,
                    x_vit=img_weak_vit,
                    return_train_features=True
                )
            momentum_inner.train(momentum_prev_mode)

        dcr_weight = self.criterion.get_dcr_weights(label, domain)
        loss_sup_cnn = (self.criterion.SupLoss(res_clean_fp32['logits_cnn'], label) * dcr_weight).mean()
        loss_sup_vit = (self.criterion.SupLoss(res_clean_fp32['logits_vit'], label) * dcr_weight).mean()
        loss_sup = loss_sup_cnn + loss_sup_vit

        # ===================================================================
        # 🌟 终极解耦对比学习模块 (Decoupled Contrastive Learning)
        # ===================================================================
        raw_target_cnn = res_momentum['proj_cnn'].detach().float()
        raw_target_vit = res_momentum['proj_vit'].detach().float()

        z_target_cnn_inst = F.normalize(raw_target_cnn, dim=-1)
        z_target_vit_inst = F.normalize(raw_target_vit, dim=-1)
        self._update_class_prototypes(z_target_cnn_inst, z_target_vit_inst, label)
        z_target_cnn_grade = self.sample_queue_mean(raw_target_cnn, label, self.queue_cnn, self.class_proto_cnn)
        z_target_vit_grade = self.sample_queue_mean(raw_target_vit, label, self.queue_vit, self.class_proto_vit)

        p_online_cnn = F.normalize(res_clean_fp32.get('pred_cnn', res_clean_fp32['proj_cnn']), dim=-1)
        p_online_vit = F.normalize(res_clean_fp32.get('pred_vit', res_clean_fp32['proj_vit']), dim=-1)

        loss_inst_cnn2vit = - (p_online_cnn * z_target_vit_inst).sum(dim=-1).mean()
        loss_inst_vit2cnn = - (p_online_vit * z_target_cnn_inst).sum(dim=-1).mean()
        loss_instance = 0.5 * (loss_inst_cnn2vit + loss_inst_vit2cnn)

        loss_grade_cnn2vit = - (p_online_cnn * z_target_vit_grade).sum(dim=-1).mean()
        loss_grade_vit2cnn = - (p_online_vit * z_target_cnn_grade).sum(dim=-1).mean()
        loss_grade = 0.5 * (loss_grade_cnn2vit + loss_grade_vit2cnn)

        current_epoch = self.epoch
        max_epochs = self.cfg.EPOCHS if self.cfg.EPOCHS > 0 else 100
        progress = current_epoch / max_epochs

        if progress < 0.2:
            lambda_grade = 0.0
        else:
            lambda_grade = 0.5 * ((progress - 0.2) / 0.8)

        loss_contrastive = loss_instance + lambda_grade * loss_grade
        self.dequeue_and_enqueue(z_target_cnn_inst, z_target_vit_inst, label)

        loss_main = loss_sup
        loss_dict = {
            'loss': loss_main.item(),
            'lr_backbone': self.optimizer.param_groups[0]['lr'],
            'lr_head': self.optimizer.param_groups[1]['lr'],
            'sup_cnn': loss_sup_cnn.item(),
            'sup_vit': loss_sup_vit.item(),
            'dcr_weight_max': dcr_weight.max().item(),
            'loss_instance': loss_instance.item(),
            'loss_grade': loss_grade.item(),
            'lambda_grade': lambda_grade,
        }

        # ==========================================
        # 真正完美的双向互蒸馏 (Mutual KD)
        # 1) KD 只做分布对齐，不再混入 true_dist，避免与 supervised CE 双重计算
        # 2) 纯 KL + T^2 缩放，保证温度补偿的数学一致性
        # ==========================================
        kd_temp = 2.0
        with torch.no_grad():
            vit_soft = F.softmax(res_momentum['logits_vit'].detach() / kd_temp, dim=1)
            cnn_soft = F.softmax(res_momentum['logits_cnn'].detach() / kd_temp, dim=1)

        log_prob_cnn = F.log_softmax(res_clean_fp32['logits_cnn'] / kd_temp, dim=1)
        loss_kd_cnn = F.kl_div(log_prob_cnn, vit_soft, reduction='batchmean') * (kd_temp ** 2)

        log_prob_vit = F.log_softmax(res_clean_fp32['logits_vit'] / kd_temp, dim=1)
        loss_kd_vit = F.kl_div(log_prob_vit, cnn_soft, reduction='batchmean') * (kd_temp ** 2)

        warmup_epochs = self.warmup_epochs
        max_epochs = getattr(getattr(self.cfg, 'OPTIM', object()), 'MAX_EPOCH', self.max_epochs)
        current_epoch = self.epoch
        if current_epoch < warmup_epochs:
            kd_weight = 0.0
        else:
            denom = max(1, max_epochs - warmup_epochs)
            progress = min(1.0, max(0.0, (current_epoch - warmup_epochs) / denom))
            kd_weight = math.exp(-5.0 * (1.0 - progress) ** 2)

        lambda_contrastive = 1.0

        # ===================================================================
        # 🚀 零参数正交解耦 (Zero-Param Orthogonal Disentanglement) - 专家修订版
        # 理论基础：强制 DINOv3 的 DRTs(TIA空间) 离开 Spatial Tokens(TRA空间)
        # ===================================================================
        tra_cls = res_clean_fp32['feat_vit']
        tia_cls = res_clean_fp32['tia_cls']

        # 1. 锚点设为不可撼动，阻断梯度，保护病理空间不被反向污染
        tra_anchor_norm = F.normalize(tra_cls, dim=-1).unsqueeze(2).detach()

        # 2. TIA 空间保留梯度，强迫其去参数化地拟合那些不属于疾病的相机特征
        tia_norm = F.normalize(tia_cls, dim=-1).unsqueeze(1)

        # 3. 计算同一样本在两个独立网络分支中的余弦相似度
        cos_sim_ortho = torch.bmm(tia_norm, tra_anchor_norm).squeeze(2).squeeze(1)

        # 4. 平方惩罚
        loss_ortho = torch.mean(cos_sim_ortho ** 2)

        probe_tia_norm = tia_cls.norm(dim=-1).mean().item()
        probe_spatial_norm = res_clean_fp32['spatial_tokens'].norm(dim=-1).mean().item()

        lambda_ortho = 1.5
        total_loss = loss_main + lambda_contrastive * loss_contrastive + kd_weight * (loss_kd_cnn + loss_kd_vit) + lambda_ortho * loss_ortho

        with torch.no_grad():
            pred_cnn_classes = res_clean_fp32['logits_cnn'].argmax(dim=1)
            pred_vit_classes = res_momentum['logits_vit'].argmax(dim=1)
            unique_classes_cnn = len(torch.unique(pred_cnn_classes))
            unique_classes_vit = len(torch.unique(pred_vit_classes))

            # 监控熵时使用真实温度 T=1.0，避免日志被蒸馏温度扭曲
            vit_probs_for_log = F.softmax(res_momentum['logits_vit'].detach(), dim=1)
            ema_vit_entropy = compute_entropy(vit_probs_for_log)
            cnn_probs = F.softmax(res_clean_fp32['logits_cnn'].detach(), dim=1)
            cnn_entropy = compute_entropy(cnn_probs)

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)

        grad_norm_backbone, grad_norm_head = self._compute_grad_norms()
        torch.nn.utils.clip_grad_norm_(self.backbone_params, max_norm=2.0)
        torch.nn.utils.clip_grad_norm_(self.head_params, max_norm=5.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            trainable_names = {k for k, v in network_inner.named_parameters() if v.requires_grad}
            network_state = network_inner.state_dict()
            momentum_state = momentum_inner.state_dict()
            for key in network_state.keys():
                if 'num_batches_tracked' in key:
                    momentum_state[key].copy_(network_state[key])
                    continue
                if key not in trainable_names and 'running_mean' not in key and 'running_var' not in key:
                    continue
                param_q = network_state[key]
                param_k = momentum_state[key]
                if param_k.is_floating_point():
                    param_k.data.mul_(self.m).add_(param_q.data, alpha=1.0 - self.m)
                else:
                    param_k.data.copy_(param_q.data)

        loss_dict['loss_kd_cnn'] = loss_kd_cnn.item() if kd_weight > 0 else 0.0
        loss_dict['loss_kd_vit'] = loss_kd_vit.item() if kd_weight > 0 else 0.0
        loss_dict['kd_weight'] = kd_weight
        loss_dict['probe_grad_backbone'] = grad_norm_backbone
        loss_dict['probe_grad_head'] = grad_norm_head
        loss_dict['probe_ent_ema_teacher'] = ema_vit_entropy
        loss_dict['probe_ent_cnn_student'] = cnn_entropy
        loss_dict['probe_unique_cls_cnn'] = unique_classes_cnn
        loss_dict['probe_unique_cls_vit'] = unique_classes_vit
        loss_dict['loss_ortho'] = loss_ortho.item()
        loss_dict['probe_tia_norm'] = probe_tia_norm
        loss_dict['probe_spatial_norm'] = probe_spatial_norm
        loss_dict['loss'] = total_loss.item()
        if torch.cuda.is_available():
            loss_dict['peak_vram_MB'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return loss_dict

    def update_epoch(self, epoch):
        self.epoch = epoch
        if epoch < self.warmup_epochs:
            warmup_ratio = (epoch + 1) / self.warmup_epochs
            self.optimizer.param_groups[0]['lr'] = self.base_lr_backbone * warmup_ratio
            self.optimizer.param_groups[1]['lr'] = self.base_lr_head * warmup_ratio
        else:
            self.scheduler.step()
        if hasattr(self.criterion, 'update_alpha'):
            self.criterion.update_alpha(epoch)

        if torch.cuda.is_available():
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
            logging.info(
                f"[Memory Monitor] Epoch {epoch} | "
                f"Peak Allocated: {peak_alloc_mb:.0f} MB | "
                f"Peak Reserved (nvidia-smi): {peak_reserved_mb:.0f} MB"
            )
            torch.cuda.reset_peak_memory_stats()
        return epoch

    def validate(self, val_loader, test_loader, writer):
        metrics_cnn_val, _ = algorithm_validate(self, val_loader, writer, self.epoch, 'val_cnn')
        metrics_cnn_test, _ = algorithm_validate(self, test_loader, writer, self.epoch, 'test_cnn')
        val_auc_cnn = metrics_cnn_val['auc']
        test_auc_cnn = metrics_cnn_test['auc']
        if self.epoch == self.cfg.EPOCHS:
            self.epoch += 1
        return val_auc_cnn, test_auc_cnn

    def predict(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], dtype=x.dtype, device=x.device)
        else:
            mask = mask.to(x.dtype)
            if mask.shape[-2:] != x.shape[-2:]:
                mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')

        x_masked = x * mask
        x_vit = F.interpolate(x_masked, size=(1024, 1024), mode='bilinear', align_corners=False)
        x_cnn = F.interpolate(x_masked, size=(1024, 1024), mode='bilinear', align_corners=False)
        return self.network(x_cnn=x_cnn, x_vit=x_vit)

    def save_model(self, log_path, source='best'):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            logging.info(f"Saving {source} model...")
            if hasattr(self.network, 'module'):
                state_dict = self.network.module.state_dict()
            else:
                state_dict = self.network.state_dict()

            if source == 'cnn':
                torch.save(state_dict, os.path.join(log_path, 'best_model_cnn.pth'))
                torch.save({'queue_cnn': self.queue_cnn, 'queue_vit': self.queue_vit, 'queue_labels': self.queue_labels, 'queue_ptr': self.queue_ptr}, os.path.join(log_path, 'queue_state_cnn.pth'))
            elif source == 'vit':
                torch.save(state_dict, os.path.join(log_path, 'best_model_vit.pth'))
                torch.save({'queue_cnn': self.queue_cnn, 'queue_vit': self.queue_vit, 'queue_labels': self.queue_labels, 'queue_ptr': self.queue_ptr}, os.path.join(log_path, 'queue_state_vit.pth'))
            else:
                torch.save(state_dict, os.path.join(log_path, 'best_model.pth'))
                torch.save({'queue_cnn': self.queue_cnn, 'queue_vit': self.queue_vit, 'queue_labels': self.queue_labels, 'queue_ptr': self.queue_ptr}, os.path.join(log_path, 'queue_state.pth'))

        if dist.is_initialized():
            dist.barrier()

    def renew_model(self, log_path, source='best'):
        if source == 'cnn':
            filename = 'best_model_cnn.pth'
        elif source == 'vit':
            filename = 'best_model_vit.pth'
        else:
            filename = 'best_model.pth'
        net_path = os.path.join(log_path, filename)
        if os.path.exists(net_path):
            state_dict = torch.load(net_path, map_location='cpu')
            if hasattr(self.network, 'module'):
                self.network.module.load_state_dict(state_dict)
            else:
                self.network.load_state_dict(state_dict)
            logging.info(f"✅ Model renewed from {filename}")
        else:
            logging.warning(f"⚠️ Could not find {filename}, skipping load.")
