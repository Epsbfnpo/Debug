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

        self.network = models.get_net(cfg)
        self.network_ema = deepcopy(self.network)
        for param in self.network_ema.parameters():
            param.requires_grad = False

        # --- 维度自动探测 ---
        # ResNet50 SideNet 融合后通常是 2048 (或 512, 取决于 fusion)
        # 建议直接写死或者运行一次 forward 探测，这里默认 2048
        dim_in = 2048
        feat_dim = 512

        self.projector = nn.Sequential(nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim, bias=False), nn.BatchNorm1d(feat_dim, affine=False))

        self.predictor = nn.Sequential(nn.Linear(feat_dim, feat_dim, bias=False), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim))

        self.optimizer = torch.optim.AdamW([{"params": self.network.parameters(), 'fix_lr': False}, {"params": self.projector.parameters(), 'fix_lr': False}, {"params": self.predictor.parameters(), 'fix_lr': True},], lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

        self.K = cfg.MOCO_QUEUE_K
        self.register_buffer("queue", torch.randn(self.K, feat_dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(self.K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.num_positive = cfg.POSITIVE if hasattr(cfg, 'POSITIVE') else 0

        self.criterion = DahLoss_Siam_Fastmoco_v0(cfg=cfg, beta=cfg.GDRNET.BETA, max_iteration=cfg.EPOCHS, training_domains=cfg.DATASET.SOURCE_DOMAINS, temperature=cfg.GDRNET.TEMPERATURE,scaling_factor=cfg.GDRNET.SCALING_FACTOR, fastmoco=True)

        self.fundusAug = get_post_FundusAug(cfg)

        self.split_num = 2
        self.combs = 3
        self.global_step = 0

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
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand([N, L], device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = x * mask.unsqueeze(-1)
        x_masked_c = x * (1 - mask.unsqueeze(-1))

        return mask.unsqueeze(-1), (1 - mask.unsqueeze(-1))

    def _local_split(self, x):     # NxCxHxW --> 4NxCx(H/2)x(W/2)
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)
        x = torch.cat(xs, dim=0)
        return x

    # --- SIAM 辅助工具 ---
    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
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
                    choice = torch.multinomial(torch.ones_like(pos).type(torch.FloatTensor), self.num_positive, replacement=True)
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

    def update(self, minibatch):
        self.global_step += 1
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        image_new, image_ori = self.img_process(image, mask, self.fundusAug)

        curr_device = image_new.device

        if next(self.projector.parameters()).device != curr_device:
            self.projector.to(curr_device)

        if next(self.predictor.parameters()).device != curr_device:
            self.predictor.to(curr_device)

        if next(self.network_ema.parameters()).device != curr_device:
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

        logits_ori, feats_ori = self.network(image_ori, force_resize=True)
        z_ori = self.projector(feats_ori)
        p_ori = self.predictor(z_ori)

        logits_new, feats_new = self.network(image_new, force_resize=True)
        z_new = self.projector(feats_new)
        p_new = self.predictor(z_new)

        logits_masked, feats_masked = self.network(image_new_masked, force_resize=True)
        z_masked = self.projector(feats_masked)
        p_masked = self.predictor(z_masked)

        z_ori = nn.functional.normalize(z_ori, dim=-1)
        z_new = nn.functional.normalize(z_new, dim=-1)
        p_ori = nn.functional.normalize(p_ori, dim=-1)
        p_new = nn.functional.normalize(p_new, dim=-1)
        z_masked = nn.functional.normalize(z_masked, dim=-1)
        p_masked = nn.functional.normalize(p_masked, dim=-1)

        with torch.no_grad():
            _, feats_new_ema = self.network_ema(image_new, force_resize=True)
            z_new_ema = self.projector(feats_new_ema)
            z_new_ema = nn.functional.normalize(z_new_ema, dim=-1)

        _, z1_split_feats = self.network(x1_splits, force_resize=False)
        _, z2_split_feats = self.network(x2_splits, force_resize=False)

        z1_split_list = list(z1_split_feats.split(image_ori.size(0), dim=0))
        z2_split_list = list(z2_split_feats.split(image_new.size(0), dim=0))

        z1_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z1_split_list, r=self.combs)))), dim=0)
        z2_orthmix_ = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z2_split_list, r=self.combs)))), dim=0)

        z1_orthmix = self.projector(z1_orthmix_)
        z2_orthmix = self.projector(z2_orthmix_)
        p1_orthmix = self.predictor(z1_orthmix)
        p2_orthmix = self.predictor(z2_orthmix)

        z1_orthmix = nn.functional.normalize(z1_orthmix, dim=-1)
        z2_orthmix = nn.functional.normalize(z2_orthmix, dim=-1)
        p1_orthmix = nn.functional.normalize(p1_orthmix, dim=-1)
        p2_orthmix = nn.functional.normalize(p2_orthmix, dim=-1)

        z1_sup = self.sample_target(z_ori.detach(), label)
        z2_sup = self.sample_target(z_new_ema.detach(), label)

        feature_list = [feats_ori, feats_new, z_ori, z_new_ema, z1_sup, z2_sup, p_ori, p_new, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix]

        masked_list = [z_masked, None, p_masked, None]

        loss, loss_dict_iter = self.criterion([logits_new, logits_ori, logits_masked, logits_ori], feature_list, masked_list, label, domain, None)

        loss.backward()
        self.optimizer.step()
        self.update_ema_model()
        self.dequeue_and_enqueue(z_new_ema.detach(), label)

        return loss_dict_iter

    def predict(self, x):
        logits, _ = self.network(x)
        return logits

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