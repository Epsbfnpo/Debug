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

ALGORITHMS = ['ERM', 'GDRNet', 'GREEN', 'CABNet', 'MixupNet', 'MixStyleNet', 'Fishr', 'DRGen', 'CASS_GDRNet']

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
        self.criterion = GDRNetLoss_Integrated(max_iteration=cfg.EPOCHS, training_domains=cfg.DATASET.SOURCE_DOMAINS, beta=cfg.GDRNET.BETA, gamma=cfg.GDRNET.GAMMA)
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

<<<<<<< codex/replace-cass_gdrnet-class-with-pure-cnn-ad0pn5
        # ================== 1. 引入双塔网络 (CNN + ViT) ==================
        self.network = DualTowerGDRNet(cfg)

        # ================== 2. 引入 CASS 的非对称 Predictor ==================
        proj_dim = 1024
        self.predictor_cnn = nn.Sequential(
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
        self.predictor_vit = nn.Sequential(
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

        # 优化器包含双塔网络和 Predictor 的参数
        trainable_params = list(self.network.parameters()) +                            list(self.predictor_cnn.parameters()) +                            list(self.predictor_vit.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=cfg.LEARNING_RATE, weight_decay=0.0001)

        # ================== 3. 引入 CASS 队列 (Queue) ==================
        self.K = 1024
        self.num_positive = getattr(cfg, 'POSITIVE', 4) # 动态多项式采样参数
        self.register_buffer("queue_cnn", torch.randn(self.K, proj_dim))
        self.register_buffer("queue_vit", torch.randn(self.K, proj_dim))
        self.queue_cnn = F.normalize(self.queue_cnn, dim=-1)
        self.queue_vit = F.normalize(self.queue_vit, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(self.K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # ================== 4. 引入集成损失 (CE + CASS) ==================
        self.criterion = GDRNetLoss_Integrated(
            training_domains=cfg.DATASET.SOURCE_DOMAINS,
            beta=cfg.GDRNET.BETA
        )

        self.fundusAug = get_post_FundusAug(cfg)
        self.scaler = torch.cuda.amp.GradScaler()

    @torch.no_grad()
    def dequeue_and_enqueue(self, proj_cnn, proj_vit, labels):
        batch_size = proj_cnn.shape[0]
        ptr = int(self.queue_ptr)
        replace_idx = torch.arange(ptr, ptr + batch_size).cuda() % self.K
        self.queue_cnn[replace_idx, :] = proj_cnn
        self.queue_vit[replace_idx, :] = proj_vit
        self.queue_labels[replace_idx] = labels
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def sample_target(self, current_features, labels, target_queue):
        neighbor = []
        for i, label in enumerate(labels):
            pos = torch.where(self.queue_labels == label)[0]
            if len(pos) != 0:
                if self.num_positive > 0:
                    weights = torch.ones_like(pos).float()
                    choice = torch.multinomial(weights, self.num_positive, replacement=True)
                    idx = pos[choice]
                    neighbor.append(target_queue[idx].mean(0))
                else:
                    neighbor.append(target_queue[pos].mean(0))
            else:
                neighbor.append(current_features[i])
        targets = torch.stack(neighbor, dim=0)
        return F.normalize(targets, dim=-1)

=======
        # 1. 纯粹的 CNN (通过 cfg 加载 ResNet50)
        self.network = models.get_net(cfg)
        self.classifier = models.get_classifier(self.network.out_features(), cfg)

        # 2. 仅优化 CNN 和 分类器 的参数
        trainable_params = list(self.network.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=cfg.LEARNING_RATE, weight_decay=0.0001)

        # 3. 保留基础的数据增强模块
        self.fundusAug = get_post_FundusAug(cfg)
        self.scaler = torch.cuda.amp.GradScaler()

>>>>>>> main
    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

<<<<<<< codex/replace-cass_gdrnet-class-with-pure-cnn-ad0pn5
        # 基础数据增强 (仅使用 strong aug)
=======
        # 1. 基础数据增强 (仅保留原图的 fundusAug 增强，去掉强弱增强对比和 Masking)
>>>>>>> main
        img_strong, mask_strong = self.fundusAug['post_aug1'](image.clone(), mask.clone())
        img_strong = img_strong * mask_strong
        img_strong = self.fundusAug['post_aug2'](img_strong).contiguous()

<<<<<<< codex/replace-cass_gdrnet-class-with-pure-cnn-ad0pn5
        with torch.amp.autocast('cuda'):
            # 1. 双塔前向传播 (此版本没有 Feature Fusion 桥接)
            res_dict = self.network(img_strong)

            proj_cnn = res_dict['proj_cnn'].float()
            proj_vit = res_dict['proj_vit'].float()
            logits_cnn = res_dict['logits_cnn'].float()
            logits_vit = res_dict['logits_vit'].float()

            # 2. 经过 Predictor 获取交叉预测特征
            pred_cnn = self.predictor_cnn(proj_cnn)
            pred_vit = self.predictor_vit(proj_vit)

            res_fp32 = {
                'proj_cnn': proj_cnn,
                'proj_vit': proj_vit,
                'pred_cnn': pred_cnn,
                'pred_vit': pred_vit,
                'logits_cnn': logits_cnn,
                'logits_vit': logits_vit,
            }

            # 3. 跨模态动态 Queue 采样 (ViT 查 CNN 队列，CNN 查 ViT 队列)
            target_vit_for_cnn = self.sample_target(proj_vit.detach(), label, self.queue_vit)
            target_cnn_for_vit = self.sample_target(proj_cnn.detach(), label, self.queue_cnn)
            target_dict = {'target_vit': target_vit_for_cnn, 'target_cnn': target_cnn_for_vit}

            # 4. 计算监督损失和 CASS 损失 (无 KD，无 FastMoCo)
            loss_main, loss_dict_inner, dcr_weight = self.criterion(res_fp32, target_dict, label, domain)

            total_loss = loss_main

        # 5. 反向传播与优化
        self.scaler.scale(total_loss).backward()
=======
        # 2. 纯 CNN 前向传播与交叉熵损失 (CE Loss)
        with torch.amp.autocast('cuda'):
            features = self.network(img_strong)
            logits = self.classifier(features)
            loss = F.cross_entropy(logits, label)

        # 3. 反向传播与优化
        self.scaler.scale(loss).backward()
>>>>>>> main
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

<<<<<<< codex/replace-cass_gdrnet-class-with-pure-cnn-ad0pn5
        # 6. 入队更新
        self.dequeue_and_enqueue(proj_cnn.detach(), proj_vit.detach(), label)

        return loss_dict_inner
=======
        loss_dict = {'loss': loss.item()}
        return loss_dict
>>>>>>> main

    def update_epoch(self, epoch):
        self.epoch = epoch
        return epoch

    def validate(self, val_loader, test_loader, writer):
<<<<<<< codex/replace-cass_gdrnet-class-with-pure-cnn-ad0pn5
        # 双塔模型，分别验证 CNN 分支和 ViT 分支的表现
        metrics_val_cnn, _ = algorithm_validate(self, val_loader, writer, self.epoch, 'val_cnn')
        metrics_test_cnn, _ = algorithm_validate(self, test_loader, writer, self.epoch, 'test_cnn')

        metrics_val_vit, _ = algorithm_validate(self, val_loader, writer, self.epoch, 'val_vit')
        metrics_test_vit, _ = algorithm_validate(self, test_loader, writer, self.epoch, 'test_vit')

        # 记录两个分支中表现最好的
        val_auc = max(metrics_val_cnn['auc'], metrics_val_vit['auc'])
        test_auc = max(metrics_test_cnn['auc'], metrics_test_vit['auc'])
=======
        # 纯 CNN 只需要验证一条分支
        metrics_val, _ = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
        metrics_test, _ = algorithm_validate(self, test_loader, writer, self.epoch, 'test')

        val_auc = metrics_val['auc']
        test_auc = metrics_test['auc']
>>>>>>> main

        if self.epoch == self.cfg.EPOCHS:
            self.epoch += 1
        if self.epoch > self.cfg.EPOCHS:
            logging.info("=" * 30)
<<<<<<< codex/replace-cass_gdrnet-class-with-pure-cnn-ad0pn5
            logging.info(f"🚀 FINAL RESULTS - Step 2: + CASS (Epoch {self.epoch - 1})")
            logging.info(f"✅ CNN Branch >> Val AUC: {metrics_val_cnn['auc']:.4f} | Test AUC: {metrics_test_cnn['auc']:.4f}")
            logging.info(f"✅ ViT Branch >> Val AUC: {metrics_val_vit['auc']:.4f} | Test AUC: {metrics_test_vit['auc']:.4f}")
=======
            logging.info(f"🚀 FINAL RESULTS - Pure CNN (Epoch {self.epoch - 1})")
            logging.info(f"✅ Baseline >> Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")
>>>>>>> main
            logging.info("=" * 30)

        return val_auc, test_auc

    def predict(self, x):
<<<<<<< codex/replace-cass_gdrnet-class-with-pure-cnn-ad0pn5
        # 验证预测时，采用双塔的 Logits 平均作为 Ensemble 结果
        res = self.network(x)
        return (res['logits_cnn'] + res['logits_vit']) / 2.0
=======
        # 预测时直接走 CNN
        return self.classifier(self.network(x))
>>>>>>> main

    def save_model(self, log_path, source='best'):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
<<<<<<< codex/replace-cass_gdrnet-class-with-pure-cnn-ad0pn5
            logging.info("Saving Step 2 CASS model...")
            state_dict = self.network.module.state_dict() if hasattr(self.network, 'module') else self.network.state_dict()
            torch.save(state_dict, os.path.join(log_path, 'best_model.pth'))
            # 注意：双塔模型的分类器已经集成在 DualTowerGDRNet 内部，无需单独保存

    def renew_model(self, log_path, source='best'):
        net_path = os.path.join(log_path, 'best_model.pth')
=======
            logging.info("Saving pure CNN model...")
            state_dict = self.network.module.state_dict() if hasattr(self.network, 'module') else self.network.state_dict()
            torch.save(state_dict, os.path.join(log_path, 'best_model.pth'))
            torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))

    def renew_model(self, log_path, source='best'):
        net_path = os.path.join(log_path, 'best_model.pth')
        cls_path = os.path.join(log_path, 'best_classifier.pth')
>>>>>>> main
        if os.path.exists(net_path):
            state_dict = torch.load(net_path, map_location='cpu')
            if hasattr(self.network, 'module'):
                self.network.module.load_state_dict(state_dict)
            else:
                self.network.load_state_dict(state_dict)
<<<<<<< codex/replace-cass_gdrnet-class-with-pure-cnn-ad0pn5
=======
            self.classifier.load_state_dict(torch.load(cls_path, map_location='cpu'))
>>>>>>> main
            logging.info(f"✅ Model renewed from {net_path}")
