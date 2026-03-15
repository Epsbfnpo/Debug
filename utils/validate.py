import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging
from tqdm import tqdm
import numpy as np

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

def calculate_metrics_numpy(real_labels, real_preds, real_outputs):
    # 总体指标
    acc = accuracy_score(real_labels, real_preds)
    f1 = f1_score(real_labels, real_preds, average='macro')
    try:
        auc_ovo = roc_auc_score(real_labels, real_outputs, average='macro', multi_class='ovo')
    except ValueError:
        auc_ovo = 0.0

    # 每个类别的 OvR 指标
    num_classes = real_outputs.shape[1] if real_outputs.ndim > 1 else 0
    per_class_metrics = {}
    if num_classes > 0:
        f1_per_class = f1_score(real_labels, real_preds, average=None, labels=range(num_classes))

        for c in range(num_classes):
            y_true_c = (real_labels == c).astype(int)
            y_pred_c = (real_preds == c).astype(int)

            acc_c = accuracy_score(y_true_c, y_pred_c)

            try:
                if len(np.unique(y_true_c)) > 1:
                    auc_c = roc_auc_score(y_true_c, real_outputs[:, c])
                else:
                    auc_c = 0.0
            except ValueError:
                auc_c = 0.0

            per_class_metrics[c] = {'acc': acc_c, 'f1': f1_per_class[c], 'auc': auc_c}

    return acc, f1, auc_ovo, per_class_metrics

def algorithm_validate(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    device = next(algorithm.network.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        stream_data = {'default': {'loss': 0.0, 'samples': 0, 'preds': [], 'labels': [], 'outputs': [], 'indices': []}}
        if rank == 0:
            loader_bar = tqdm(data_loader, desc=f"Validating ({val_type}) [DDP]", leave=False, dynamic_ncols=True)
        else:
            loader_bar = data_loader
        is_dual_stream = False
        for batch in loader_bar:
            try:
                if len(batch) == 5:
                    image = batch[0]
                    label = batch[2]
                    index = batch[4]
                else:
                    image = batch[0]
                    label = batch[1]
                    index = batch[3]
            except IndexError as e:
                print(f"[Rank {rank}] Error unpacking batch! Len: {len(batch)}")
                raise e
            image = image.to(device)
            label = label.to(device).long()
            index = index.to(device).long()
            output = algorithm.predict(image)
            if isinstance(output, dict):
                if not is_dual_stream:
                    is_dual_stream = True
                    del stream_data['default']
                    stream_data['cnn'] = {'loss': 0.0, 'samples': 0, 'preds': [], 'labels': [], 'outputs': [], 'indices': []}
                    stream_data['vit'] = {'loss': 0.0, 'samples': 0, 'preds': [], 'labels': [], 'outputs': [], 'indices': []}
                for stream_name, logits in [('cnn', output['logits_cnn']), ('vit', output['logits_vit'])]:
                    batch_loss = criterion(logits, label).item()
                    stream_data[stream_name]['loss'] += batch_loss * image.size(0)
                    stream_data[stream_name]['samples'] += image.size(0)
                    _, pred = torch.max(logits, 1)
                    output_sf = softmax(logits)
                    stream_data[stream_name]['preds'].append(pred)
                    stream_data[stream_name]['labels'].append(label)
                    stream_data[stream_name]['outputs'].append(output_sf)
                    stream_data[stream_name]['indices'].append(index)
                current_loss_display = criterion(output['logits_cnn'], label).item()
            else:
                logits = output
                batch_loss = criterion(logits, label).item()
                stream_data['default']['loss'] += batch_loss * image.size(0)
                stream_data['default']['samples'] += image.size(0)
                _, pred = torch.max(logits, 1)
                output_sf = softmax(logits)
                stream_data['default']['preds'].append(pred)
                stream_data['default']['labels'].append(label)
                stream_data['default']['outputs'].append(output_sf)
                stream_data['default']['indices'].append(index)
                current_loss_display = batch_loss
            if rank == 0 and isinstance(loader_bar, tqdm):
                loader_bar.set_postfix({'loss': f'{current_loss_display:.4f}'})
        final_metrics = {}
        for stream_name, data in stream_data.items():
            if len(data['preds']) > 0:
                local_preds = torch.cat(data['preds'])
                local_labels = torch.cat(data['labels'])
                local_outputs = torch.cat(data['outputs'])
                local_indices = torch.cat(data['indices'])
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
                total_loss_tensor = torch.tensor([data['loss']], device=device)
                total_samples_tensor = torch.tensor([data['samples']], device=device)
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
                if total_samples_tensor.item() > 0:
                    final_loss = total_loss_tensor.item() / total_samples_tensor.item()
                else:
                    final_loss = 0.0
            else:
                all_preds, all_labels, all_outputs, all_indices = local_preds, local_labels, local_outputs, local_indices
                if data['samples'] > 0:
                    final_loss = data['loss'] / data['samples']
                else:
                    final_loss = 0.0
            metrics_stream = {'auc': 0.0, 'acc': 0.0, 'f1': 0.0, 'loss': final_loss}
            num_classes = 0
            if rank == 0:
                all_indices_cpu = all_indices.cpu().numpy()
                if len(all_indices_cpu) > 0:
                    _, unique_mask = np.unique(all_indices_cpu, return_index=True)
                    real_preds = all_preds.cpu().numpy()[unique_mask]
                    real_labels = all_labels.cpu().numpy()[unique_mask]
                    real_outputs = all_outputs.cpu().numpy()[unique_mask]

                    acc, f1, auc_ovo, per_class_metrics = calculate_metrics_numpy(real_labels, real_preds, real_outputs)
                    num_classes = real_outputs.shape[1]
                else:
                    acc, f1, auc_ovo = 0.0, 0.0, 0.0
                    per_class_metrics = {}
                    num_classes = 0
                metrics_stream = {'auc': auc_ovo, 'acc': acc, 'f1': f1, 'loss': final_loss}

                for c in range(num_classes):
                    if c in per_class_metrics:
                        metrics_stream[f'class_{c}_auc'] = per_class_metrics[c]['auc']
                        metrics_stream[f'class_{c}_acc'] = per_class_metrics[c]['acc']
                        metrics_stream[f'class_{c}_f1'] = per_class_metrics[c]['f1']

                prefix = f"{val_type}"
                if is_dual_stream:
                    prefix = f"{val_type}/{stream_name.upper()}"
                    log_str = f'[{stream_name.upper()}] {val_type} - Epoch: {epoch}, Loss: {final_loss:.4f}, Acc: {acc:.4f}, AUC: {auc_ovo:.4f}, F1: {f1:.4f}'
                else:
                    log_str = f'{val_type} - Epoch: {epoch}, Loss: {final_loss:.4f}, Acc: {acc:.4f}, AUC: {auc_ovo:.4f}, F1: {f1:.4f}'

                if val_type == 'test' and num_classes > 0:
                    detail_str = " | " + " ".join([
                        f"C{c}(AUC:{per_class_metrics[c]['auc']:.4f} ACC:{per_class_metrics[c]['acc']:.4f} F1:{per_class_metrics[c]['f1']:.4f})"
                        for c in range(num_classes)
                    ])
                    log_str += detail_str

                logging.info(log_str)

                if writer is not None:
                    writer.add_scalar(f'info/{prefix}_accuracy', acc, epoch)
                    writer.add_scalar(f'info/{prefix}_loss', final_loss, epoch)
                    writer.add_scalar(f'info/{prefix}_auc_ovo', auc_ovo, epoch)
                    writer.add_scalar(f'info/{prefix}_f1', f1, epoch)
            if is_dual_stream:
                final_metrics[f'{stream_name}_auc'] = metrics_stream.get('auc', 0.0)
                final_metrics[f'{stream_name}_acc'] = metrics_stream.get('acc', 0.0)
                final_metrics[f'{stream_name}_f1'] = metrics_stream.get('f1', 0.0)
                for c in range(num_classes):
                    final_metrics[f'{stream_name}_class_{c}_auc'] = metrics_stream.get(f'class_{c}_auc', 0.0)
                    final_metrics[f'{stream_name}_class_{c}_acc'] = metrics_stream.get(f'class_{c}_acc', 0.0)
                    final_metrics[f'{stream_name}_class_{c}_f1'] = metrics_stream.get(f'class_{c}_f1', 0.0)
            else:
                final_metrics = metrics_stream
        if is_dual_stream:
            final_metrics['auc'] = final_metrics.get('cnn_auc', 0.0)
            final_metrics['acc'] = final_metrics.get('cnn_acc', 0.0)
            final_metrics['f1'] = final_metrics.get('cnn_f1', 0.0)
            final_metrics['loss'] = metrics_stream.get('loss', 0.0)
        algorithm.train()
        return final_metrics, final_metrics.get('loss', 0.0)
    return None, 0.0
