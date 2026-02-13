import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging
from tqdm import tqdm
import numpy as np


def gather_tensor(tensor):
    """
    辅助函数：收集所有 GPU 上的 tensor。
    """
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


def algorithm_validate(algorithm, data_loader, writer, epoch, val_type):
    """
    支持多卡并行的验证函数。
    所有卡都会运行 inference，然后通过 all_gather 汇总结果，由 Rank 0 计算指标。
    """
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # 获取当前设备
    device = next(algorithm.network.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0

    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)

        local_loss = 0.0
        local_samples = 0

        list_preds = []
        list_labels = []
        list_outputs = []
        list_indices = []

        if rank == 0:
            loader_bar = tqdm(data_loader, desc=f"Validating ({val_type}) [DDP]", leave=False, dynamic_ncols=True)
        else:
            loader_bar = data_loader

        for batch in loader_bar:
            # ================= [修正点] =================
            # 验证集/测试集 dataset __getitem__ 返回的是: (image, label, domain, index)
            # 索引分别是 0, 1, 2, 3
            # 之前的 batch[4] 是错误的，因为没有 mask
            # ============================================
            try:
                if len(batch) == 5:  # 兼容 Train 模式 (虽然 validate 一般不用)
                    image = batch[0]
                    # mask = batch[1] (验证不需要 mask)
                    label = batch[2]
                    # domain = batch[3]
                    index = batch[4]
                else:  # 正常的 Val/Test 模式 (4个元素)
                    image = batch[0]
                    label = batch[1]
                    # domain = batch[2] (验证一般不需要 domain，如果需要可加上)
                    index = batch[3]
            except IndexError as e:
                # 打印错误信息方便调试
                print(f"[Rank {rank}] Error unpacking batch! Len: {len(batch)}")
                raise e

            image = image.to(device)
            label = label.to(device).long()
            index = index.to(device).long()

            output = algorithm.predict(image)
            batch_loss = criterion(output, label).item()

            local_loss += batch_loss * image.size(0)
            local_samples += image.size(0)

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            list_preds.append(pred)
            list_labels.append(label)
            list_outputs.append(output_sf)
            list_indices.append(index)

            if rank == 0 and isinstance(loader_bar, tqdm):
                loader_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

        # 1. 拼接本地结果
        if len(list_preds) > 0:
            local_preds = torch.cat(list_preds)
            local_labels = torch.cat(list_labels)
            local_outputs = torch.cat(list_outputs)
            local_indices = torch.cat(list_indices)
        else:
            # 防止某个卡数据为空 (极少见但可能)
            local_preds = torch.tensor([], device=device)
            local_labels = torch.tensor([], device=device)
            local_outputs = torch.tensor([], device=device)
            local_indices = torch.tensor([], device=device)

        # 2. 分布式汇总
        if dist.is_initialized():
            all_preds = gather_tensor(local_preds)
            all_labels = gather_tensor(local_labels)
            all_outputs = gather_tensor(local_outputs)
            all_indices = gather_tensor(local_indices)

            total_loss_tensor = torch.tensor([local_loss], device=device)
            total_samples_tensor = torch.tensor([local_samples], device=device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

            # 防止除以零
            if total_samples_tensor.item() > 0:
                final_loss = total_loss_tensor.item() / total_samples_tensor.item()
            else:
                final_loss = 0.0
        else:
            all_preds, all_labels, all_outputs, all_indices = local_preds, local_labels, local_outputs, local_indices
            if local_samples > 0:
                final_loss = local_loss / local_samples
            else:
                final_loss = 0.0

        # 3. 数据去重与指标计算
        if rank == 0:
            all_indices_cpu = all_indices.cpu().numpy()

            if len(all_indices_cpu) > 0:
                _, unique_mask = np.unique(all_indices_cpu, return_index=True)

                real_preds = all_preds.cpu().numpy()[unique_mask]
                real_labels = all_labels.cpu().numpy()[unique_mask]
                real_outputs = all_outputs.cpu().numpy()[unique_mask]

                acc = accuracy_score(real_labels, real_preds)
                f1 = f1_score(real_labels, real_preds, average='macro')

                try:
                    auc_ovo = roc_auc_score(real_labels, real_outputs, average='macro', multi_class='ovo')
                except ValueError:
                    auc_ovo = 0.0
            else:
                # 没有任何数据
                acc, f1, auc_ovo = 0.0, 0.0, 0.0

            if val_type in ['val', 'test']:
                if writer is not None:
                    writer.add_scalar(f'info/{val_type}_accuracy', acc, epoch)
                    writer.add_scalar(f'info/{val_type}_loss', final_loss, epoch)
                    writer.add_scalar(f'info/{val_type}_auc_ovo', auc_ovo, epoch)
                    writer.add_scalar(f'info/{val_type}_f1', f1, epoch)

                logging.info(
                    f'{val_type} - epoch: {epoch}, loss: {final_loss:.4f}, acc: {acc:.4f}, auc: {auc_ovo:.4f}, F1: {f1:.4f}.')

            return auc_ovo, final_loss

        else:
            return 0.0, 0.0

    algorithm.train()
    return 0.0, 0.0