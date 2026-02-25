from .resnet import resnet18, resnet50, resnet101
import torch
import math
import random
import torch.nn as nn
from torch.nn import Module
import torchvision.models
import torch.nn.functional as F
import numpy as np
import copy
from copy import deepcopy
from collections import deque
import logging
from transformers import AutoModel, AutoConfig
from .lora_utils import inject_lora_dinov3
import types

def important_token_selection(key_layer, value_layer, attention_probs, token_ratio=0.2):
    B, N, C = key_layer.shape
    num_keep = int(N * token_ratio)
    importance_scores = attention_probs.mean(dim=1).mean(dim=-2)
    topk_indices = importance_scores.topk(num_keep, dim=-1).indices
    gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)
    k_selected = torch.gather(key_layer, dim=1, index=gather_indices)
    v_selected = torch.gather(value_layer, dim=1, index=gather_indices)
    return k_selected, v_selected

class BridgeModule(nn.Module):
    def __init__(self, cnn_dim, vit_dim, project_dim=512, num_heads=8, token_ratio=0.5):
        super().__init__()
        self.token_ratio = token_ratio
        self.num_heads = num_heads
        self.proj_cnn = nn.Linear(cnn_dim, project_dim)
        self.proj_vit = nn.Linear(vit_dim, project_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=project_dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(project_dim)
        self.proj_back = nn.Linear(project_dim, cnn_dim)

    def forward(self, feat_cnn, feat_vit, attn_map_vit=None):
        B, C, H, W = feat_cnn.shape
        cnn_flat = feat_cnn.flatten(2).transpose(1, 2)
        query = self.proj_cnn(cnn_flat)
        if attn_map_vit is not None:
            key_vit, val_vit = important_token_selection(feat_vit, feat_vit, attn_map_vit, self.token_ratio)
        else:
            key_vit, val_vit = feat_vit, feat_vit
        key = self.proj_vit(key_vit)
        value = self.proj_vit(val_vit)
        attn_out, _ = self.cross_attn(query, key, value)
        out = torch.tanh(self.gate) * self.norm(attn_out)
        out = self.proj_back(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return feat_cnn + out

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SoftLabelGCN(nn.Module):
    def __init__(self, cfg):
        super(SoftLabelGCN, self).__init__()
        self.cnn = eval(cfg.BACKBONE)(pretrained = True, in_channel = 3, avgpool=True)
        in_features = self.cnn._out_features
        self.cfg = cfg
        hidden_dim = int(in_features / 2)
        self.gcn1 = GraphConvolution(cfg.DATASET.NUM_CLASSES, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, in_features)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.)
        self.register_parameter('adj', nn.Parameter(torch.tensor(self.get_gcn_adj(), dtype=torch.float)))
        self.register_buffer('adj_mask', torch.tensor([[0,0,0,0,0], [1,0,0,0,0], [1,1,0,0,0], [1,1,1,0,0], [1,1,1,1,0]], dtype=torch.float))
        self.register_buffer('inp', torch.tensor(self.get_gcn_inp(), dtype=torch.float))
        self.register_buffer('diag', torch.eye(cfg.DATASET.NUM_CLASSES, dtype=torch.float))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.cnn_linear = nn.Linear(in_features, cfg.DATASET.NUM_CLASSES)
        self.remove_gcngate = False
    
    def get_gcn_inp(self):
        inp = np.eye(self.cfg.DATASET.NUM_CLASSES)
        return inp

    def get_gcn_adj(self):
        adj = 0.2 * np.eye(self.cfg.DATASET.NUM_CLASSES, self.cfg.DATASET.NUM_CLASSES, 1) + 0.2 * np.eye(self.cfg.DATASET.NUM_CLASSES, self.cfg.DATASET.NUM_CLASSES, -1)
        return adj

    def forward(self, input):
        temp = self.adj * self.adj_mask.detach()
        temp = temp + temp.t()
        a = self.relu(temp) + self.diag.detach()
        D = torch.pow(a.sum(1).float(), -0.5)
        D = torch.diag(D)
        A = torch.matmul(torch.matmul(a, D).t(), D)
        x = self.gcn1(self.inp.detach(), A)
        x = self.leaky_relu(x)
        x = self.gcn2(x, A.detach())
        x = x.transpose(0, 1)
        cnn_x = self.cnn(input)
        cnn_x = cnn_x.view(cnn_x.size(0), -1)
        x = torch.matmul(cnn_x.detach(), x)
        cnn_x = self.cnn_linear(cnn_x)
        x = self.sigmoid(x)
        if self.remove_gcngate: x = 0
        return x * cnn_x + cnn_x
    
    def get_config_optim(self, lr_cnn, lr_gcn, lr_adj):
        return [{'params': self.cnn.parameters(), 'lr': lr_cnn}, {'params': self.cnn_linear.parameters(), 'lr': lr_cnn}, {'params': self.gcn1.parameters(), 'lr': lr_gcn}, {'params': self.gcn2.parameters(), 'lr': lr_gcn}, {'params': self.adj, 'lr': lr_adj},]

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types
    
    def logsumexp_2d(self, tensor):
        tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
        s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
        outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
        return outputs
    
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            x = x.view(x.size(0), x.size(1), 1, -1)
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                lse_pool = self.logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale, torch.sigmoid( channel_att_sum )

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale, scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out, _ = self.ChannelGate(x)
        if not self.no_spatial:
            x_out, scale = self.SpatialGate(x_out)
        return x_out

class CABlock(nn.Module):
    def __init__(self, input, classes, k):
        super(CABlock, self).__init__()
        self.classes = classes
        self.k = k
        self.conv1 = nn.Conv2d(input, classes*k, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout2d(0.1)
        self.BN = nn.BatchNorm2d(classes*k)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AvgPool2d((1, 1))

    def forward(self, x):
        x1 = self.relu(self.BN(self.conv1(x)))
        x2 = self.dropout(x1)
        x2 = self.maxpool(x2).squeeze().squeeze()
        x2 = x2.view(x2.shape[0], self.classes, self.k)
        weight = torch.mean(x2, dim=-1)
        weight = weight.unsqueeze(dim=-1).unsqueeze(dim=-1)
        n,c,h,w = x1.size()
        res = x1.reshape(n, self.classes, self.k, h, w).sum(2)
        new = weight * res
        attention = new.reshape(n, 1, self.classes, h, w).sum(2)
        return attention * x

class CABNet(nn.Module):
    def __init__(self, cfg) -> None:
        super(CABNet, self).__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout(0.0)
        self.model = eval(cfg.BACKBONE)(pretrained = True, in_channel = 3)
        in_features = self.model._out_features
        self.global_attention = nn.Sequential(CBAM(in_features), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(in_features, in_features//2))
        self.category_attention = CABlock(in_features//2, cfg.DATASET.NUM_CLASSES, 5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._out_features = in_features//2

    def out_features(self):
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features
    
    def forward(self, x):
        feature = self.dropout(self.model(x))
        feature = self.global_attention(feature).unsqueeze(dim=2).unsqueeze(dim=3)
        feature = self.category_attention(feature)
        feature = (self.pool(feature)).squeeze()
        return feature

class MixupNet(nn.Module):
    def __init__(self, cfg) -> None:
        super(MixupNet, self).__init__()
        self.cfg = cfg
        self.model = eval(cfg.BACKBONE)(pretrained = True, in_channel = 3, avgpool=False)
        in_features = self.model._out_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._out_features = in_features

    def out_features(self):
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1), 1, -1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})")

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x
        if random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        if self.mix == "random":
            perm = torch.randperm(B)
        elif self.mix == "crossdomain":
            perm = torch.arange(B - 1, -1, -1)
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)
        return x_normed*sig_mix + mu_mix

class MixStyleNet(nn.Module):
    def __init__(self, cfg):
        super(MixStyleNet, self).__init__()
        filter = [64, 128, 256, 512, 512]
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_size = 512
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)
        self._out_features = self.feature_size
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def out_features(self):
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features
    
    def conv_layer(self, in_channel, out_channel, index):
        if index < 3:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True), nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),)
        else:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True), nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True), nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),)
        return conv_block

    def forward(self, x):
        g_block1 = self.block1(x)
        g_block1 = self.mixstyle(g_block1)
        g_block2 = self.block2(g_block1)
        g_block2 = self.mixstyle(g_block2)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        feature = self.block5(g_block4)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        return feature

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class FishrNet(torch.nn.Module):
    def __init__(self, cfg, network=None):
        super(FishrNet, self).__init__()
        if network is None:
            network = torchvision.models.resnet50(pretrained=True)
        self.network = network
        self._out_features = 2048
        del self.network.fc
        self.network.fc = Identity()
        self.dropout = nn.Dropout(cfg.DROP_OUT)
        self.freeze_bn()

    def forward(self, x):
        return self.dropout(self.network(x))

    def out_features(self):
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features
    
    def train(self, mode=True):
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class SWADBase:
    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):
        raise NotImplementedError()

    def get_final_model(self):
        raise NotImplementedError()

class LossValley(SWADBase):
    def __init__(self, evaluator, n_converge, n_tolerance, tolerance_ratio, **kwargs):
        self.evaluator = evaluator
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio
        self.converge_Q = deque(maxlen=n_converge)
        self.smooth_Q = deque(maxlen=n_tolerance)
        self.final_model = None
        self.converge_step = None
        self.dead_valley = False
        self.threshold = None

    def get_smooth_loss(self, idx):
        smooth_loss = min([model.end_loss for model in list(self.smooth_Q)[idx:]])
        return smooth_loss

    @property
    def is_converged(self):
        return self.converge_step is not None

    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):
        if self.dead_valley:
            return
        frozen = copy.deepcopy(segment_swa)
        frozen.end_loss = val_loss
        self.converge_Q.append(frozen)
        self.smooth_Q.append(frozen)
        if not self.is_converged:
            if len(self.converge_Q) < self.n_converge:
                return
            min_idx = np.argmin([model.end_loss for model in self.converge_Q])
            untilmin_segment_swa = self.converge_Q[min_idx]
            if min_idx == 0:
                self.converge_step = self.converge_Q[0].end_step
                self.final_model = AveragedModel(untilmin_segment_swa)
                th_base = np.mean([model.end_loss for model in self.converge_Q])
                self.threshold = th_base * (1.0 + self.tolerance_ratio)
                if self.n_tolerance < self.n_converge:
                    for i in range(self.n_converge - self.n_tolerance):
                        model = self.converge_Q[1 + i]
                        self.final_model.update_parameters(model, start_step=model.start_step, end_step=model.end_step)
                elif self.n_tolerance > self.n_converge:
                    converge_idx = self.n_tolerance - self.n_converge
                    Q = list(self.smooth_Q)[: converge_idx + 1]
                    start_idx = 0
                    for i in reversed(range(len(Q))):
                        model = Q[i]
                        if model.end_loss > self.threshold:
                            start_idx = i + 1
                            break
                    for model in Q[start_idx + 1 :]:
                        self.final_model.update_parameters(model, start_step=model.start_step, end_step=model.end_step)
                print(f"Model converged at step {self.converge_step}, "f"Start step = {self.final_model.start_step}; "f"Threshold = {self.threshold:.6f}, ")
            return
        if self.smooth_Q[0].end_step < self.converge_step:
            return
        min_vloss = self.get_smooth_loss(0)
        if min_vloss > self.threshold:
            self.dead_valley = False
            print(f"Valley is dead at step {self.final_model.end_step}")
            return
        model = self.smooth_Q[0]
        self.final_model.update_parameters(model, start_step=model.start_step, end_step=model.end_step)

    def get_final_model(self):
        if not self.is_converged:
            logging.warning("Requested final model, but model is not yet converged; return last model instead")
            return self.converge_Q[-1]
        if not self.dead_valley:
            self.smooth_Q.popleft()
            while self.smooth_Q:
                smooth_loss = self.get_smooth_loss(0)
                if smooth_loss > self.threshold:
                    break
                segment_swa = self.smooth_Q.popleft()
                self.final_model.update_parameters(segment_swa, step=segment_swa.end_step)
        return self.final_model

class AveragedModel(Module):
    def __init__(self, model, device=None, avg_fn=None, rm_optimizer=False):
        super(AveragedModel, self).__init__()
        self.start_step = -1
        self.end_step = -1
        if isinstance(model, AveragedModel):
            model = model.module
        self.module = deepcopy(model)
        if rm_optimizer:
            for k, v in vars(self.module).items():
                if isinstance(v, torch.optim.Optimizer):
                    setattr(self.module, k, None)
        self.module = self.module.cuda()
        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.module.predict(*args, **kwargs)

    @property
    def network(self):
        return self.module.network

    def update_parameters(self, model, step=None, start_step=None, end_step=None):
        if isinstance(model, AveragedModel):
            model = model.module
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device)))
        self.n_averaged += 1
        if step is not None:
            if start_step is None:
                start_step = step
            if end_step is None:
                end_step = step
        if start_step is not None:
            if self.n_averaged == 1:
                self.start_step = start_step
        if end_step is not None:
            self.end_step = end_step

    def clone(self):
        clone = copy.deepcopy(self.module)
        clone.optimizer = clone.new_optimizer(clone.network.parameters())
        return clone

class DINOv3Wrapper(nn.Module):
    def __init__(self, local_path, lora_r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        print(f"🚀 [DINOv3+LoRA] Loading from local path: {local_path}")
        try:
            self.config = AutoConfig.from_pretrained(local_path, local_files_only=True)
            self.config.output_hidden_states = True
            self.config.output_attentions = False
            self._out_features = self.config.hidden_size
            self.model = AutoModel.from_pretrained(local_path, config=self.config, local_files_only=True)
            self.model.gradient_checkpointing_enable()
            for param in self.model.parameters():
                param.requires_grad = False
            print("🧊 [DINOv3] All base parameters frozen.")
            inject_lora_dinov3(self.model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            print(f"🔥 [LoRA Injected] Trainable parameters: {len(trainable_params)}")
        except Exception as e:
            print(f"❌ Error loading DINOv3: {e}")
            raise e

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs

    @property
    def out_features(self):
        return self._out_features


class DualTowerGDRNet(nn.Module):
    def __init__(self, cfg):
        super(DualTowerGDRNet, self).__init__()
        self.cnn = resnet50(pretrained=True)
        if hasattr(self.cnn, 'out_features'):
            self.cnn_dim = self.cnn.out_features()
        else:
            self.cnn_dim = 2048
        default_dinov3_path = "/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/checkpoints/dinov3_vitb16"
        dinov3_path = getattr(cfg.GDRNET, 'DINOV3_PATH', default_dinov3_path)
        lora_r = getattr(cfg.GDRNET, 'LORA_R', 8)
        lora_alpha = getattr(cfg.GDRNET, 'LORA_ALPHA', 16)
        lora_dropout = getattr(cfg.GDRNET, 'LORA_DROPOUT', 0.0)
        self.vit = DINOv3Wrapper(local_path=dinov3_path, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.vit_dim = self.vit.out_features
        self.target_layers = [3, 6, 9]
        self.hooked_attns = {}
        def make_hook(layer_idx):
            def new_forward(attn_self, *args, **kwargs):
                kwargs['output_attentions'] = True
                outputs = attn_self._original_forward(*args, **kwargs)
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    attn_self.current_attn_map = outputs[1]
                return outputs
            return new_forward
        hook_count = 0
        for name, module in self.vit.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) >= 12:
                for l_idx in self.target_layers:
                    if hasattr(module[l_idx], 'attention'):
                        target_attention = module[l_idx].attention
                        target_attention._original_forward = target_attention.forward
                        target_attention.forward = types.MethodType(make_hook(l_idx), target_attention)
                        self.hooked_attns[l_idx] = target_attention
                        hook_count += 1
                break
        if hook_count == len(self.target_layers):
            print(f"✅ 成功注入 {hook_count} 个 Hook：已精准拦截 DINOv3 第 {self.target_layers} 层的 Attention Map。")
        else:
            print("⚠️ 警告：多层 Hook 注入不完整，请检查 DINOv3 结构。")
        self.bridge1 = BridgeModule(cnn_dim=256, vit_dim=self.vit_dim, project_dim=256, token_ratio=0.5)
        self.bridge2 = BridgeModule(cnn_dim=512, vit_dim=self.vit_dim, project_dim=512, token_ratio=0.4)
        self.bridge3 = BridgeModule(cnn_dim=1024, vit_dim=self.vit_dim, project_dim=512, token_ratio=0.3)
        proj_dim = 1024
        self.projector_cnn = nn.Sequential(nn.Linear(self.cnn_dim, self.cnn_dim), nn.BatchNorm1d(self.cnn_dim), nn.ReLU(inplace=True), nn.Linear(self.cnn_dim, proj_dim))
        self.projector_vit = nn.Sequential(nn.Linear(self.vit_dim, self.vit_dim), nn.BatchNorm1d(self.vit_dim), nn.ReLU(inplace=True), nn.Linear(self.vit_dim, proj_dim))
        self.classifier_cnn = nn.Linear(self.cnn_dim, cfg.DATASET.NUM_CLASSES)
        self.classifier_vit = nn.Linear(self.vit_dim, cfg.DATASET.NUM_CLASSES)

    def forward(self, x_cnn, x_vit=None):
        res = {}
        if x_vit is None:
            img_for_cnn = x_cnn
            img_for_vit = x_cnn
        else:
            img_for_cnn = x_cnn
            img_for_vit = x_vit
        vit_outputs = self.vit(img_for_vit)
        feat_vit_3 = vit_outputs.hidden_states[3 + 1]
        feat_vit_6 = vit_outputs.hidden_states[6 + 1]
        feat_vit_9 = vit_outputs.hidden_states[9 + 1]
        attn_3 = getattr(self.hooked_attns[3], 'current_attn_map', None)
        attn_6 = getattr(self.hooked_attns[6], 'current_attn_map', None)
        attn_9 = getattr(self.hooked_attns[9], 'current_attn_map', None)
        for l_idx in self.target_layers:
            if hasattr(self.hooked_attns[l_idx], 'current_attn_map'):
                self.hooked_attns[l_idx].current_attn_map = None
        feat_vit_final = vit_outputs.last_hidden_state[:, 0]
        x = self.cnn.conv1(img_for_cnn)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)
        x = self.bridge1(feat_cnn=x, feat_vit=feat_vit_3, attn_map_vit=attn_3)
        x = self.cnn.layer2(x)
        x = self.bridge2(feat_cnn=x, feat_vit=feat_vit_6, attn_map_vit=attn_6)
        x = self.cnn.layer3(x)
        x = self.bridge3(feat_cnn=x, feat_vit=feat_vit_9, attn_map_vit=attn_9)
        x = self.cnn.layer4(x)
        x = self.cnn.global_avgpool(x)
        feat_cnn_final = torch.flatten(x, 1)
        logits_cnn = self.classifier_cnn(feat_cnn_final)
        proj_cnn = self.projector_cnn(feat_cnn_final)
        res['logits_cnn'] = logits_cnn
        res['proj_cnn'] = proj_cnn
        logits_vit = self.classifier_vit(feat_vit_final)
        proj_vit = self.projector_vit(feat_vit_final)
        res['logits_vit'] = logits_vit
        res['proj_vit'] = proj_vit
        return res

    def extract_cnn_feature(self, x_cnn, x_vit=None):
        if x_vit is None:
            x_vit = x_cnn
        vit_outputs = self.vit(x_vit)
        feat_vit_3 = vit_outputs.hidden_states[4]
        feat_vit_6 = vit_outputs.hidden_states[7]
        feat_vit_9 = vit_outputs.hidden_states[10]
        attn_3 = getattr(self.hooked_attns[3], 'current_attn_map', None)
        attn_6 = getattr(self.hooked_attns[6], 'current_attn_map', None)
        attn_9 = getattr(self.hooked_attns[9], 'current_attn_map', None)
        for l_idx in self.target_layers:
            if hasattr(self.hooked_attns[l_idx], 'current_attn_map'):
                self.hooked_attns[l_idx].current_attn_map = None
        x = self.cnn.conv1(x_cnn)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)
        x = self.bridge1(feat_cnn=x, feat_vit=feat_vit_3, attn_map_vit=attn_3)
        x = self.cnn.layer2(x)
        x = self.bridge2(feat_cnn=x, feat_vit=feat_vit_6, attn_map_vit=attn_6)
        x = self.cnn.layer3(x)
        x = self.bridge3(feat_cnn=x, feat_vit=feat_vit_9, attn_map_vit=attn_9)
        x = self.cnn.layer4(x)
        x = self.cnn.global_avgpool(x)
        feat_cnn_final = torch.flatten(x, 1)
        return feat_cnn_final