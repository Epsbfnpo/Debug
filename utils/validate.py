import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, cohen_kappa_score
import logging
from tqdm import tqdm


def concat_all_gather(tensor):
    if not dist.is_initialized():
        return tensor

    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def algorithm_validate(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()

    is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

    local_loss = 0.0
    local_steps = 0
    prob_list = []
    target_list = []
    pred_list = []

    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)

        loader_bar = tqdm(data_loader, desc=f"Validating ({val_type})", leave=False, dynamic_ncols=True,
                          disable=not is_main_process)

        # 适配 GDRBench 返回的 9 个值
        for batch in loader_bar:
            # 安全解包：通过索引获取需要的数据
            image = batch[0].cuda()
            label = batch[4].cuda().long()
            domain = batch[7].cuda().long()

            output = algorithm.predict(image)

            # 兼容字典输出
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output

            loss = criterion(logits, label)
            prob = softmax(logits)
            _, pred = torch.max(prob, 1)

            local_loss += loss.item()
            local_steps += 1

            prob_list.append(prob)
            target_list.append(label)
            pred_list.append(pred)

    local_prob = torch.cat(prob_list, dim=0)
    local_target = torch.cat(target_list, dim=0)
    local_pred = torch.cat(pred_list, dim=0)

    if dist.is_initialized():
        global_prob = concat_all_gather(local_prob)
        global_target = concat_all_gather(local_target)
        global_pred = concat_all_gather(local_pred)

        metric_tensor = torch.tensor([local_loss, local_steps], device='cuda')
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        global_loss = metric_tensor[0] / max(metric_tensor[1], 1.0)
    else:
        global_prob = local_prob
        global_target = local_target
        global_pred = local_pred
        global_loss = local_loss / max(local_steps, 1)

    global_target_np = global_target.cpu().numpy()
    global_pred_np = global_pred.cpu().numpy()
    global_prob_np = global_prob.cpu().numpy()

    acc = accuracy_score(global_target_np, global_pred_np)
    f1 = f1_score(global_target_np, global_pred_np, average='macro')

    try:
        qwk = cohen_kappa_score(global_target_np, global_pred_np, weights='quadratic')
    except ValueError:
        qwk = 0.0

    try:
        auc_ovo = roc_auc_score(global_target_np, global_prob_np, average='macro', multi_class='ovo')
    except ValueError:
        auc_ovo = 0.0

    if is_main_process:
        if writer is not None:
            writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
            writer.add_scalar('info/{}_loss'.format(val_type), global_loss, epoch)
            writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)

    metrics = {
        'loss': global_loss,
        'acc': acc,
        'f1': f1,
        'qwk': qwk,
        'auc': auc_ovo
    }

    # [关键修复] 只返回一个 metrics 字典，不要返回元组
    return metrics