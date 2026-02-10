import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
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

        loader_bar = tqdm(data_loader, desc=f"Validating ({val_type})", leave=False, dynamic_ncols=True, disable=not is_main_process)

        for image, label, domain, _ in loader_bar:
            image = image.cuda()
            label = label.cuda().long()

            output = algorithm.predict(image)
            batch_loss = criterion(output, label).item()

            local_loss += batch_loss
            local_steps += 1

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            prob_list.append(output_sf)
            target_list.append(label)
            pred_list.append(pred)

            if is_main_process:
                loader_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

    if len(prob_list) > 0:
        local_prob = torch.cat(prob_list, dim=0)
        local_target = torch.cat(target_list, dim=0)
        local_pred = torch.cat(pred_list, dim=0)
    else:
        local_prob = torch.tensor([]).cuda()
        local_target = torch.tensor([]).cuda()
        local_pred = torch.tensor([]).cuda()

    if dist.is_initialized():
        global_prob = concat_all_gather(local_prob)
        global_target = concat_all_gather(local_target)
        global_pred = concat_all_gather(local_pred)

        metric_tensor = torch.tensor([local_loss, local_steps], dtype=torch.float32).cuda()
        dist.all_reduce(metric_tensor)
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
        auc_ovo = roc_auc_score(global_target_np, global_prob_np, average='macro', multi_class='ovo')
    except ValueError:
        auc_ovo = 0.0

    if is_main_process:
        if writer is not None:
            writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
            writer.add_scalar('info/{}_loss'.format(val_type), global_loss, epoch)
            writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)

        logging.info('{} - epoch: {}, loss: {:.4f}, acc: {:.4f}, auc: {:.4f}, F1: {:.4f}.'.format(val_type, epoch, global_loss, acc, auc_ovo, f1))

    algorithm.train()
    return auc_ovo, global_loss