import csv
import datetime
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import algorithms
from dataset.data_manager import get_dataset
from utils.args import get_args, setup_cfg
from utils.misc import MultiLossCounter, init_log, update_writer
from utils.validate import METRIC_NAMES, algorithm_validate


BEST_MODEL_METRICS = [
    'macro_f1',
    'macro_ovr_auc',
    'macro_ovo_auc',
    'weighted_ovr_auc',
    'weighted_ovo_auc',
]


def debug_log(msg, rank):
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}][Rank {rank}] {msg}", flush=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path, algorithm, optimizer, scheduler, epoch, best_performance):
    state = {
        'epoch': epoch,
        'algorithm_state': algorithm.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_performance': best_performance,
    }
    torch.save(state, path)


def load_checkpoint(path, algorithm, optimizer, scheduler):
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location='cpu')
    if 'algorithm_state' in checkpoint:
        algorithm.load_state_dict(checkpoint['algorithm_state'])
    else:
        if hasattr(algorithm.network, 'module'):
            algorithm.network.module.load_state_dict(checkpoint['model_state'])
        else:
            algorithm.network.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch'] + 1
    best_performance = checkpoint.get('best_performance', 0.0)
    return start_epoch, best_performance


def _normalize_domain_loaders(test_loader):
    if isinstance(test_loader, dict):
        return test_loader
    if isinstance(test_loader, (list, tuple)):
        return {f'test_{idx}': loader for idx, loader in enumerate(test_loader)}
    return {'test': test_loader}


def _branch_candidates(algorithm):
    if algorithm.__class__.__name__ == 'CASS_GDRNet':
        return ['cnn', 'vit']
    return ['fusion']


def _init_diagnostic_csv(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['Epoch', 'Domain', 'Branch'] + METRIC_NAMES)


def _append_diagnostic_row(csv_path, row):
    with open(csv_path, 'a', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(row)


def _save_best_metric_model(algorithm, output_dir, metric_name):
    save_path = os.path.join(output_dir, f'best_val_{metric_name}.pth')
    torch.save(algorithm.state_dict(), save_path)
    return save_path


def main():
    start_time = time.time()
    args = get_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        args.device = torch.device('cuda', args.local_rank)
        is_distributed = True
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_distributed = False

    cfg = setup_cfg(args)
    root_output = os.path.abspath(cfg.OUT_DIR)
    log_path = os.path.join(root_output, cfg.OUTPUT_PATH)
    os.makedirs(log_path, exist_ok=True)

    writer = None
    if args.local_rank in [-1, 0]:
        writer = init_log(args, cfg, log_path, 0, [0, 0, 0])
        logging.info(f"Distributed: {is_distributed}, Rank: {args.local_rank}")
        print(f"[INFO] Log Path: {log_path}")

    set_seed(cfg.SEED)

    latest_ckpt_path = os.path.join(log_path, 'latest_model.pth')
    final_ckpt_path = os.path.join(log_path, 'final_model.pth')
    diagnostic_csv_path = os.path.join(log_path, 'diagnostic_metrics.csv')

    if os.path.exists(final_ckpt_path):
        if args.local_rank in [-1, 0]:
            print(f"✅ Found {final_ckpt_path}. Training already completed.")
        if is_distributed:
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)

    debug_log("Loading datasets...", args.local_rank)
    train_loader, val_loader, test_loader, dataset_size, train_sampler = get_dataset(args, cfg)
    test_loaders = _normalize_domain_loaders(test_loader)

    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
    algorithm.to(args.device)

    if is_distributed:
        algorithm = nn.SyncBatchNorm.convert_sync_batchnorm(algorithm)
        algorithm.network = DDP(
            algorithm.network,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        if hasattr(algorithm, 'projector'):
            algorithm.projector = DDP(algorithm.projector, device_ids=[args.local_rank], output_device=args.local_rank)
        if hasattr(algorithm, 'predictor'):
            algorithm.predictor = DDP(algorithm.predictor, device_ids=[args.local_rank], output_device=args.local_rank)
        if hasattr(algorithm, 'classifier') and len(list(algorithm.classifier.parameters())) > 0:
            algorithm.classifier = DDP(algorithm.classifier, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = algorithm.optimizer
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=0.00015)

    start_epoch = 1
    best_performance = 0.0

    branch_candidates = _branch_candidates(algorithm)
    if branch_candidates == ['cnn', 'vit']:
        best_val_metrics = {f"{b}_{m}": 0.0 for b in ['cnn', 'vit'] for m in BEST_MODEL_METRICS}
        best_val_epochs = {f"{b}_{m}": 0 for b in ['cnn', 'vit'] for m in BEST_MODEL_METRICS}
    else:
        best_val_metrics = {f"{b}_{m}": 0.0 for b in branch_candidates for m in BEST_MODEL_METRICS}
        best_val_epochs = {f"{b}_{m}": 0 for b in branch_candidates for m in BEST_MODEL_METRICS}

    if os.path.exists(latest_ckpt_path):
        debug_log(f"Found {latest_ckpt_path}. Resuming training...", args.local_rank)
        try:
            start_epoch, best_performance = load_checkpoint(latest_ckpt_path, algorithm, optimizer, scheduler)
            debug_log(f"Resumed from Epoch {start_epoch}.", args.local_rank)
        except Exception as e:
            debug_log(f"Error loading checkpoint: {e}. Starting from scratch.", args.local_rank)

    if args.local_rank in [-1, 0] and not os.path.exists(diagnostic_csv_path):
        _init_diagnostic_csv(diagnostic_csv_path)

    iterator = tqdm(
        range(start_epoch - 1, cfg.EPOCHS),
        disable=(args.local_rank not in [-1, 0]),
        initial=start_epoch - 1,
        total=cfg.EPOCHS,
    )

    for i in iterator:
        epoch = i + 1
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(i)

        loss_counter = MultiLossCounter()
        algorithm.train()

        for image, mask, label, domain, img_index in train_loader:
            image = image.to(args.device)
            mask = mask.to(args.device)
            label = label.to(args.device).long()
            domain = domain.to(args.device).long()
            minibatch = [image, mask, label, domain]
            step_vals = algorithm.update(minibatch)
            loss_counter.update(step_vals, image.size(0))

        if hasattr(algorithm, 'update_epoch'):
            algorithm.update_epoch(epoch)

        epoch_losses = loss_counter.get_averages()
        if args.local_rank in [-1, 0]:
            update_writer(writer, epoch, scheduler, epoch_losses)

        scheduler.step()

        if args.local_rank in [-1, 0]:
            tracked_main = epoch_losses.get('loss', 0.0)
            save_checkpoint(latest_ckpt_path, algorithm, optimizer, scheduler, epoch, best_performance=max(best_performance, tracked_main))

        if epoch % cfg.VAL_EPOCH == 0:
            if args.local_rank in [-1, 0]:
                logging.info(f"Epoch {epoch} Validation...")

            for branch in branch_candidates:
                metrics, _ = algorithm_validate(
                    algorithm,
                    val_loader,
                    writer,
                    epoch,
                    'val',
                    branch=branch,
                )

                if args.local_rank in [-1, 0]:
                    for metric_name in BEST_MODEL_METRICS:
                        key = f"{branch}_{metric_name}"
                        current_val = metrics.get(metric_name, 0.0)
                        if current_val > best_val_metrics[key]:
                            best_val_metrics[key] = current_val
                            best_val_epochs[key] = epoch
                            _save_best_metric_model(algorithm, log_path, key)
                            logging.info(f"[{key}] improved to {current_val:.6f}, saving model")

                    best_performance = max(best_performance, metrics.get('macro_f1', 0.0))

            for test_env_name, test_env_loader in test_loaders.items():
                for branch in branch_candidates:
                    test_metrics, _ = algorithm_validate(
                        algorithm,
                        test_env_loader,
                        writer,
                        epoch,
                        f'test_{test_env_name}',
                        branch=branch,
                    )

                    if args.local_rank in [-1, 0]:
                        for metric_name, val in test_metrics.items():
                            if metric_name in METRIC_NAMES:
                                writer.add_scalar(f'Test_{test_env_name}/{branch}_{metric_name}', val, global_step=epoch)
                        row = [epoch, test_env_name, branch] + [test_metrics.get(m, 0.0) for m in METRIC_NAMES]
                        _append_diagnostic_row(diagnostic_csv_path, row)

            if args.local_rank in [-1, 0]:
                logging.info(f"Epoch {epoch} Diagnostic Testing Complete.")

            if is_distributed:
                dist.barrier()

        if args.time_limit > 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > (args.time_limit - 300):
                if args.local_rank in [-1, 0]:
                    save_checkpoint(latest_ckpt_path, algorithm, optimizer, scheduler, epoch, best_performance)
                    if writer:
                        writer.close()
                if is_distributed:
                    dist.barrier()
                    dist.destroy_process_group()
                sys.exit(0)

    debug_log("Training Finished. Starting Final Testing...", args.local_rank)

    if args.local_rank in [-1, 0]:
        save_checkpoint(final_ckpt_path, algorithm, optimizer, scheduler, cfg.EPOCHS, best_performance)
        with open(os.path.join(log_path, 'done'), 'w') as f:
            f.write(f'done, best_val_metrics={best_val_metrics}, best_val_epochs={best_val_epochs}')

    if writer:
        writer.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
