import logging
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

METRIC_NAMES = ['acc', 'macro_f1', 'macro_ovr_auc', 'macro_ovo_auc', 'weighted_ovr_auc', 'weighted_ovo_auc',]

def gather_tensor(tensor):
    world_size = dist.get_world_size()
    device = tensor.device
    local_size = torch.tensor([tensor.size(0)], device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max([x.item() for x in all_sizes])
    size_diff = max_size - local_size.item()
    if size_diff > 0:
        padding = torch.zeros((size_diff, *tensor.shape[1:]), device=device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    data_list = []
    for size, data in zip(all_sizes, gather_list):
        data_list.append(data[:size.item()])
    return torch.cat(data_list, dim=0)

def calculate_metrics_numpy(y_true, y_pred, y_prob, num_classes=5):
    metrics = {}
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    labels = list(range(num_classes))
    try:
        metrics['macro_ovr_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=labels)
        metrics['macro_ovo_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro', labels=labels)
        metrics['weighted_ovr_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted', labels=labels)
        metrics['weighted_ovo_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='weighted', labels=labels)
    except ValueError as e:
        logging.warning(f"AUC calculation failed ({e}). Setting AUC metrics to 0.")
        metrics['macro_ovr_auc'] = 0.0
        metrics['macro_ovo_auc'] = 0.0
        metrics['weighted_ovr_auc'] = 0.0
        metrics['weighted_ovo_auc'] = 0.0
    metrics['f1'] = metrics['macro_f1']
    metrics['auc'] = metrics['macro_ovo_auc']
    return metrics

def _safe_predict(algorithm, image, branch=None):
    if branch is None:
        return algorithm.predict(image)
    try:
        return algorithm.predict(image, branch=branch)
    except TypeError:
        return algorithm.predict(image)

def _extract_logits(output, branch):
    if not isinstance(output, dict):
        return output
    if branch == 'cnn' and 'logits_cnn' in output:
        return output['logits_cnn']
    if branch == 'vit' and 'logits_vit' in output:
        return output['logits_vit']
    if branch == 'fusion':
        for k in ('logits_fusion', 'logits', 'pred_fusion', 'logits_cnn'):
            if k in output:
                return output[k]
    for k in ('logits', 'logits_cnn', 'logits_vit'):
        if k in output:
            return output[k]
    raise KeyError('Cannot find logits in model output dict.')

def algorithm_validate(algorithm, data_loader, writer, epoch, val_type, branch=None):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    device = next(algorithm.network.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    total_loss = 0.0
    total_samples = 0
    preds, labels, outputs, indices = [], [], [], []
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        if rank == 0:
            loader_bar = tqdm(data_loader, desc=f"Validating ({val_type}) [DDP]", leave=False, dynamic_ncols=True)
        else:
            loader_bar = data_loader
        for batch in loader_bar:
            if len(batch) == 5:
                image = batch[0]
                label = batch[2]
                index = batch[4]
            else:
                image = batch[0]
                label = batch[1]
                index = batch[3]
            image = image.to(device)
            label = label.to(device).long()
            index = index.to(device).long()
            output = _safe_predict(algorithm, image, branch=branch)
            logits = _extract_logits(output, branch)
            batch_loss = criterion(logits, label).item()
            total_loss += batch_loss * image.size(0)
            total_samples += image.size(0)
            pred = torch.argmax(logits, dim=1)
            prob = softmax(logits)
            preds.append(pred)
            labels.append(label)
            outputs.append(prob)
            indices.append(index)
            if rank == 0 and isinstance(loader_bar, tqdm):
                loader_bar.set_postfix({'loss': f'{batch_loss:.4f}'})
    if len(preds) > 0:
        local_preds = torch.cat(preds)
        local_labels = torch.cat(labels)
        local_outputs = torch.cat(outputs)
        local_indices = torch.cat(indices)
    else:
        local_preds = torch.tensor([], device=device)
        local_labels = torch.tensor([], device=device)
        local_outputs = torch.tensor([], device=device)
        local_indices = torch.tensor([], device=device)
    if dist.is_initialized():
        all_preds = gather_tensor(local_preds)
        all_labels = gather_tensor(local_labels)
        all_outputs = gather_tensor(local_outputs)
        all_indices = gather_tensor(local_indices)
        total_loss_tensor = torch.tensor([total_loss], device=device)
        total_samples_tensor = torch.tensor([total_samples], device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        final_loss = total_loss_tensor.item() / max(total_samples_tensor.item(), 1)
    else:
        all_preds, all_labels, all_outputs, all_indices = local_preds, local_labels, local_outputs, local_indices
        final_loss = total_loss / max(total_samples, 1)
    metrics = {name: 0.0 for name in METRIC_NAMES}
    metrics['loss'] = final_loss
    metrics['auc'] = 0.0
    metrics['f1'] = 0.0
    if rank == 0:
        all_indices_cpu = all_indices.cpu().numpy()
        if len(all_indices_cpu) > 0:
            _, unique_mask = np.unique(all_indices_cpu, return_index=True)
            real_preds = all_preds.cpu().numpy()[unique_mask]
            real_labels = all_labels.cpu().numpy()[unique_mask]
            real_outputs = all_outputs.cpu().numpy()[unique_mask]
            if real_outputs.ndim == 2 and real_outputs.shape[1] > 0:
                num_classes = real_outputs.shape[1]
            else:
                num_classes = getattr(getattr(algorithm, 'cfg', None), 'DATASET', None)
                num_classes = getattr(num_classes, 'NUM_CLASSES', 5)
            metrics = calculate_metrics_numpy(real_labels, real_preds, real_outputs, num_classes=num_classes)
            metrics['loss'] = final_loss
        else:
            metrics['loss'] = final_loss
        branch_suffix = f"/{branch}" if branch else ''
        logging.info(f"{val_type}{branch_suffix} - Epoch: {epoch}, Loss: {metrics['loss']:.4f}, Acc: {metrics['acc']:.4f}, MacroF1: {metrics['macro_f1']:.4f}, MacroOVR-AUC: {metrics['macro_ovr_auc']:.4f}, MacroOVO-AUC: {metrics['macro_ovo_auc']:.4f}, WeightedOVR-AUC: {metrics['weighted_ovr_auc']:.4f}, WeightedOVO-AUC: {metrics['weighted_ovo_auc']:.4f}")
        if writer is not None:
            tag_prefix = f"{val_type}{branch_suffix}"
            writer.add_scalar(f'info/{tag_prefix}_loss', metrics['loss'], epoch)
            for name in METRIC_NAMES:
                writer.add_scalar(f'info/{tag_prefix}_{name}', metrics[name], epoch)
    algorithm.train()
    return metrics, metrics['loss']
