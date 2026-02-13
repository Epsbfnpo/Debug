from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def D(p, z, random_matrix=None, version='simplified'):
    if version == 'original':
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    elif version == 'random':
        p = torch.matmul(p, random_matrix)
        z = torch.matmul(z, random_matrix)
        return - F.cosine_similarity(p, z.detach() , dim=-1).mean()

    else:
        raise Exception


class VSLoss(nn.Module):
    def __init__(self, cls_num_list, gamma=0.3, tau=0.5, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, x, target):
        output = x / self.Delta_list + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)


class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


class LogitAdjust_KD_V1(nn.Module):
    def __init__(self, cls_num_list, tau=1, temperature=1.0, weight=None):
        super(LogitAdjust_KD_V1, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list1 = tau * torch.log(cls_p_list)

        self.m_list1 = m_list1.view(1, -1)
        self.m_list2 = m_list1.view(1, -1)

        self.weight = weight
        self.temperature = temperature

    def forward(self, x, target, label):
        x_m = x + self.m_list1
        target = target + self.m_list2
        target = F.softmax(target / self.temperature, 1).detach()
        x_m = F.softmax(x / self.temperature, 1)  # .detach()

        output_target_max, output_target_index = torch.max(F.softmax((target), dim=1).detach(), dim=1)

        return -(target[(output_target_index == label)] * torch.log(x_m[(output_target_index == label)])).sum() / target.size()[0]


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, num_classes: int, gamma: int = 1, lb_smooth: float = 0.1, size_average: bool = True, ignore_index: int = None, alpha: float = None):
        super(FocalLossWithSmoothing, self).__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._size_average = size_average
        self._ignore_index = ignore_index
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._alpha = alpha

        if self._num_classes <= 1:
            raise ValueError('The number of classes must be 2 or higher')
        if self._gamma < 0:
            raise ValueError('Gamma must be 0 or higher')
        if self._alpha is not None:
            if self._alpha <= 0 or self._alpha >= 1:
                raise ValueError('Alpha must be 0 <= alpha <= 1')

    def forward(self, logits, logits_target, label):
        logits = logits.float()
        difficulty_level = self._estimate_difficulty_level(logits, label)   ###
        logs = self._log_softmax(logits)
        loss = -torch.sum(difficulty_level * logs, dim=1)
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level

    def collect_grad(self, grad):
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * self.targets, dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets), dim=0)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)


class DahLoss(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature

        self.domain_num_dict = {'MESSIDOR': 1396, 'IDRID': 413, 'DEEPDR': 1280, 'FGADR': 1474, 'APTOS': 2930, 'RLDR': 1275,'DDR': 10018, 'EYEPACS': 28101}

        self.label_num_dict = {'MESSIDOR': [824, 218, 272, 57, 25], 'IDRID': [131, 23, 135, 71, 53], 'DEEPDR': [583, 141, 253, 227, 76], 'FGADR': [81, 177, 474, 508, 234], 'APTOS': [1438, 300, 807, 156, 229], 'RLDR': [126, 272, 747, 77, 53], 'DDR': [5012, 484, 3600, 184, 738], 'EYEPACS': [20661, 1962, 4207, 702, 569]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        valid_domains = []
        for d in self.training_domains:
            if d in self.domain_num_dict:
                valid_domains.append(d)
            else:
                print(f"⚠️ [WARNING] Domain '{d}' not found in DahLoss config! It will be ignored.")

        if len(valid_domains) == 0:
             raise RuntimeError("❌ No valid training domains found in DahLoss configuration!")

        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in valid_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in valid_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight

    def forward(self, output, features, labels, domains):

        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        features_ori, features_new = features

        loss_sup = 0

        for op_item in output:
            loss_sup += self.SupLoss(op_item, labels)

        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)

        loss_unsup = torch.mean(self.UnsupLoss(features_multi))

        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()

        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, reduction = 'mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],''at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            loss = loss.view(anchor_count, batch_size)

        return loss


class DahLoss_Siam_Fastmoco_v0(nn.Module):
    def __init__(self, cfg, max_iteration, training_domains, beta=0.8, scaling_factor=4, alpha=1.0, temperature=0.07, fastmoco=1.0) -> None:
        super(DahLoss_Siam_Fastmoco_v0, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.gamma = fastmoco
        self.cfg = cfg
        self.domain_num_dict = {'MESSIDOR': 1396, 'IDRID': 413, 'DEEPDR': 1280, 'FGADR': 1474, 'APTOS': 2930, 'RLDR': 1275, 'DDR': 10018, 'EYEPACS': 28101}

        self.label_num_dict = {'MESSIDOR': [824, 218, 272, 57, 25], 'IDRID': [131, 23, 135, 71, 53], 'DEEPDR': [583, 141, 253, 227, 76], 'FGADR': [81, 177, 474, 508, 234], 'APTOS': [1438, 300, 807, 156, 229], 'RLDR': [126, 272, 747, 77, 53], 'DDR': [5012, 484, 3600, 184, 738], 'EYEPACS': [20661, 1962, 4207, 702, 569]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature=self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')
        self.SupLoss2 = nn.CrossEntropyLoss()
        self.per_cls_weights = torch.ones(5).cuda()
        self.KdLoss = LogitAdjust_KD_V1(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]], tau=0.0)
        self.KdLoss_Focal = FocalLossWithSmoothing(cfg.DATASET.NUM_CLASSES, 2.0)

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta=0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights_v2(self, epoch):
        cls_num_list = self.label_num_dict[self.training_domains[0]]
        idx = epoch // 80
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

        return self.per_cls_weights

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight

    def _estimate_difficulty_level(self, logits, label, gamma=2., alpha=0.25):
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self.cfg.DATASET.NUM_CLASSES)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits)
        difficulty_level = alpha * torch.pow(1 - pt, gamma)
        return difficulty_level


    def forward(self, output, features, features_masked, labels, domains, random_matrix):
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        output, output_new_a, output_new_masked, output_ori_masked_c = output
        if self.gamma > -1.0:
            features_ori, features_new, z1, z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix = features
            z_new_masked, z_ori_masked_c, p_new_masked, p_ori_masked_c = features_masked

        else:
            z1, z2, z1_sup, z2_sup, p1, p2 = features

        loss_sup = 0
        self.SupLoss3 = LogitAdjust(self.label_num_dict[self.training_domains[0]], tau=0.6)
        self.SupLoss4 = VSLoss(self.label_num_dict[self.training_domains[0]], gamma=0.1, tau=0.3)

        if self.cfg.TRANSFORM.FREQ:
            loss_sup += 0.5 * self.SupLoss(output, labels)
            loss_sup += 0.5 * self.SupLoss(output_new_a, labels)
        else:
            loss_sup += 1.0 * self.SupLoss(output, labels)
            loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        temperature = 1.0

        loss_sup_mask = 0
        if self.cfg.TRANSFORM.FREQ:
            loss_sup_mask += 0.5 * self.KdLoss(output_new_masked, output.detach(), labels)
            loss_sup_mask += 0.5 * self.KdLoss(output_ori_masked_c, output.detach(), labels)

        else:
            alpha_t = self.alpha * self.cfg.SMOOTH
            alpha_t = max(0, alpha_t)

            targets_numpy = labels.cpu().detach().numpy()
            identity_matrix = torch.eye(self.cfg.DATASET.NUM_CLASSES)
            targets_one_hot = identity_matrix[targets_numpy]

            soft_output = (alpha_t * targets_one_hot).to('cuda') + ((1 - alpha_t) * F.softmax(output.detach(), dim=1))
            soft_output_rb = (alpha_t * targets_one_hot).to('cuda') + ((1 - alpha_t) * F.softmax(output_new_masked, dim=1))
            loss_sup_mask += - 1.0 * (soft_output / temperature * F.log_softmax(output_new_masked / temperature, 1)).sum() / output.size()[0]

        SSL_loss = -0.5 * (torch.sum(p1 * z2.detach(), dim=-1) + torch.sum(p2 * z1.detach(), dim=-1))
        Sup_loss = -0.5 * (torch.sum(p1 * z2_sup, dim=-1) + torch.sum(p2 * z1_sup, dim=-1))
        loss_siam = 0.5 * SSL_loss.mean() + 0.5 * Sup_loss.mean()

        if isinstance(features_ori, (list, tuple)):
            feat_ori_use = features_ori[-1]
            feat_new_use = features_new[-1]
        else:
            feat_ori_use = features_ori
            feat_new_use = features_new

        features_multi = torch.stack([feat_ori_use, feat_new_use], dim=1)
        features_multi = F.normalize(features_multi, p=2, dim=2)

        loss_unsup = torch.mean(self.UnsupLoss(features_multi))

        if self.gamma > -1.0:
            SSL_loss_fastmoco = 0.0
            Sup_loss_fastmoco = 0.0

            calc_version = 'random' if random_matrix is not None else 'simplified'

            for i in range(len(p1_orthmix)):
                p1_orthmix_, p2_orthmix_ = p1_orthmix[i], p2_orthmix[i]

                if self.cfg.DG_MODE == 'DG':
                    SSL_loss_fastmoco += D(p1_orthmix_, z2, random_matrix, version=calc_version) / 2 + D(p2_orthmix_, z1, random_matrix, version=calc_version) / 2
                    Sup_loss_fastmoco += D(p1_orthmix_, z2_sup, random_matrix, version=calc_version) / 2 + D(p2_orthmix_, z1_sup, random_matrix, version=calc_version) / 2
                else:
                    SSL_loss_fastmoco += D(p1_orthmix_, z2, random_matrix, version=calc_version) / 2 + D(p2_orthmix_, z1, random_matrix, version=calc_version) / 2
                    Sup_loss_fastmoco += D(p1_orthmix_, z2_sup, random_matrix, version=calc_version) / 2 + D(p2_orthmix_, z1_sup, random_matrix, version=calc_version) / 2

            loss_siam_fastmoco1 = 0.5 * SSL_loss_fastmoco / len(p1_orthmix)
            loss_siam_fastmoco2 = 0.5 * Sup_loss_fastmoco / len(p1_orthmix)
            loss_siam_fastmoco = 0.5 * SSL_loss_fastmoco / len(p1_orthmix) + 0.5 * Sup_loss_fastmoco / len(p1_orthmix)
            w_sup = self.cfg.GDRNET.LAMBDA_SUP
            w_mask = self.cfg.GDRNET.LAMBDA_MASKED
            w_siam = self.cfg.GDRNET.LAMBDA_SIAM

            loss = loss_sup * w_sup + loss_sup_mask * w_mask + loss_siam_fastmoco * w_siam
        else:
            print(" Not FastMOCO")
            loss = loss_sup + loss_siam * self.alpha

        if self.cfg.DG_MODE == 'DG':
            loss_dict['loss'] = loss.item()
            loss_dict['loss_sup'] = loss_sup.item()
            loss_dict['loss_sup_mask'] = loss_sup_mask.item()
            loss_dict['loss_siam_fastmoco1'] = loss_siam_fastmoco1.item()
            loss_dict['loss_siam_fastmoco2'] = loss_siam_fastmoco2.item()
        else:
            loss_dict['loss'] = loss.item()
            loss_dict['loss_sup'] = loss_sup.item()
            loss_dict['loss_siam'] = loss_siam.item()
            loss_dict['loss_siam_fastmoco'] = loss_siam_fastmoco.item()

        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration

        return self.alpha