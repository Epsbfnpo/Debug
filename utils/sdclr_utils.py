import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class Mask(object):
    def __init__(self, model, no_reset=False):
        self.model = model
        if not no_reset:
            self.reset()

    @property
    def sparsity(self):
        prunableTensors = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        if not prunableTensors:
            return 0.0

        unpruned = torch.sum(torch.stack([torch.sum(v) for v in prunableTensors]))
        total = torch.sum(torch.stack([torch.sum(torch.ones_like(v)) for v in prunableTensors]))
        return 1 - unpruned.float() / total.float()

    def magnitudePruning(self, magnitudePruneFraction):
        weights = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                weights.append(module.weight.clone().cpu().detach().numpy())

        if not weights:
            return

        self.reset()

        prunableTensors = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        weight_vector = np.concatenate([v.flatten() for v in weights])
        number_of_weights_to_prune = int(np.ceil(magnitudePruneFraction * len(weight_vector)))
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = (torch.abs(module.weight) >= threshold).float()

    def reset(self):
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = torch.ones_like(module.weight)