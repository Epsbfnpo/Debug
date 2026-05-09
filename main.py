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
from utils.validate import algorithm_validate

# =========================
# Final model selection setup
# =========================
SELECTOR_BRANCH = 'fusion'
SELECTOR_AUC_NAME = 'weighted_ovr_auc'
SELECTOR_METRICS = ['acc', 'macro_f1', SELECTOR_AUC_NAME]

# Non-linear validation-only selector (fitted from the previous 6-source experiments)
SELECTOR_STD = 0.20

SELECTOR_MEAN = {
    'acc': 0.731301,
    'macro_f1': 0.569152,
    'weighted_ovr_auc': 0.908833,
}

SELECTOR_SCALE = {
    'acc': 0.095509,
    'macro_f1': 0.159586,
    'weighted_ovr_auc': 0.055108,
}

SELECTOR_CENTERS = [
    (1.125544,  0.848749,  1.124460),
    (0.260701,  0.465256,  0.719799),
    (-0.031419, 0.255338,  0.199003),
    (-0.744444, -0.099330, -0.051414),
    (0.255466,  0.405100,  0.050205),
    (-0.545509, 0.195182, -0.971427),
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

def save_checkpoint(path, algorithm, optimizer, scheduler, epoch, best_performance,
                    best_selector_score=-float('inf'),
                    best_selector_epoch=-1,
                    best_selector_metrics=None):
    state = {
        'epoch': epoch,
        'algorithm_state': algorithm.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_performance': best_performance,
        'best_selector_score': best_selector_score,
        'best_selector_epoch': best_selector_epoch,
        'best_selector_metrics': best_selector_metrics or {},
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
    best_selector_score = checkpoint.get('best_selector_score', -float('inf'))
    best_selector_epoch = checkpoint.get('best_selector_epoch', -1)
    best_selector_metrics = checkpoint.get('best_selector_metrics', {})
    return start_epoch, best_performance, best_selector_score, best_selector_epoch, best_selector_metrics

def _normalize_domain_loaders(test_loader):
    if isinstance(test_loader, dict):
        return test_loader
    if isinstance(test_loader, (list, tuple)):
        return {f'test_{idx}': loader for idx, loader in enumerate(test_loader)}
    return {'test': test_loader}

def _safe_standardize(value, mean, std, eps=1e-12):
    return (value - mean) / max(std, eps)


def compute_selector_score(metrics):
    """
    Use validation metrics only:
        x = acc
        y = macro_f1
        z = weighted_ovr_auc

    Score(epoch) = sum_k exp(-||[x,y,z]-center_k||^2 / tau^2)
    """
    x = _safe_standardize(metrics['acc'], SELECTOR_MEAN['acc'], SELECTOR_SCALE['acc'])
    y = _safe_standardize(metrics['macro_f1'], SELECTOR_MEAN['macro_f1'], SELECTOR_SCALE['macro_f1'])
    z = _safe_standardize(
        metrics['weighted_ovr_auc'],
        SELECTOR_MEAN['weighted_ovr_auc'],
        SELECTOR_SCALE['weighted_ovr_auc'],
    )

    tau_sq = SELECTOR_STD ** 2
    score = 0.0
    for a, b, c in SELECTOR_CENTERS:
        dist_sq = (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2
        score += np.exp(-dist_sq / tau_sq)

    return float(score)


def save_selector_checkpoint(path, algorithm, epoch, selector_score, selector_metrics):
    state = {
        'epoch': epoch,
        'selector_score': float(selector_score),
        'selector_metrics': selector_metrics,
        'algorithm_state': algorithm.state_dict(),
    }
    torch.save(state, path)


def load_selector_checkpoint(path, algorithm):
    checkpoint = torch.load(path, map_location='cpu')
    algorithm.load_state_dict(checkpoint['algorithm_state'])
    return (
        checkpoint.get('epoch', -1),
        checkpoint.get('selector_score', None),
        checkpoint.get('selector_metrics', {}),
    )

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
        algorithm.network = DDP(algorithm.network, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,)
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
    best_selector_score = -float('inf')
    best_selector_epoch = -1
    best_selector_metrics = {}
    best_selector_path = os.path.join(log_path, 'best_val_selector_model.pth')
    if os.path.exists(latest_ckpt_path):
        debug_log(f"Found {latest_ckpt_path}. Resuming training...", args.local_rank)
        try:
            start_epoch, best_performance, best_selector_score, best_selector_epoch, best_selector_metrics = \
                load_checkpoint(latest_ckpt_path, algorithm, optimizer, scheduler)
            debug_log(f"Resumed from Epoch {start_epoch}.", args.local_rank)
        except Exception as e:
            debug_log(f"Error loading checkpoint: {e}. Starting from scratch.", args.local_rank)
    iterator = tqdm(range(start_epoch - 1, cfg.EPOCHS), disable=(args.local_rank not in [-1, 0]), initial=start_epoch - 1, total=cfg.EPOCHS,)
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
        if epoch % cfg.VAL_EPOCH == 0:
            if args.local_rank in [-1, 0]:
                logging.info(f"Epoch {epoch} Validation...")

            val_metrics, _ = algorithm_validate(
                algorithm,
                val_loader,
                writer,
                epoch,
                'val',
                branch=SELECTOR_BRANCH,
            )
            selector_score = compute_selector_score(val_metrics)

            if args.local_rank in [-1, 0]:
                if selector_score > best_selector_score:
                    best_selector_score = selector_score
                    best_selector_epoch = epoch
                    best_selector_metrics = {
                        metric_name: float(val_metrics[metric_name])
                        for metric_name in SELECTOR_METRICS
                    }
                    save_selector_checkpoint(
                        best_selector_path,
                        algorithm,
                        epoch,
                        best_selector_score,
                        best_selector_metrics,
                    )

                best_performance = max(best_performance, val_metrics.get('macro_f1', 0.0))

            if is_distributed:
                dist.barrier()

        if args.local_rank in [-1, 0]:
            tracked_main = epoch_losses.get('loss', 0.0)
            save_checkpoint(
                latest_ckpt_path,
                algorithm,
                optimizer,
                scheduler,
                epoch,
                best_performance=max(best_performance, tracked_main),
                best_selector_score=best_selector_score,
                best_selector_epoch=best_selector_epoch,
                best_selector_metrics=best_selector_metrics,
            )
        if args.time_limit > 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > (args.time_limit - 300):
                if args.local_rank in [-1, 0]:
                    save_checkpoint(
                        latest_ckpt_path,
                        algorithm,
                        optimizer,
                        scheduler,
                        epoch,
                        best_performance,
                        best_selector_score=best_selector_score,
                        best_selector_epoch=best_selector_epoch,
                        best_selector_metrics=best_selector_metrics,
                    )
                    if writer:
                        writer.close()
                if is_distributed:
                    dist.barrier()
                    dist.destroy_process_group()
                sys.exit(0)
    debug_log("Training Finished. Starting Final Model Selection + Final Testing...", args.local_rank)

    if args.local_rank in [-1, 0]:
        if best_selector_epoch < 0:
            raise RuntimeError("No best validation model was selected. Please check validation scheduling.")

        logging.info(f"Selected epoch = {best_selector_epoch} for final testing.")

    if is_distributed:
        dist.barrier()

    selected_epoch, selected_score, selected_metrics = load_selector_checkpoint(best_selector_path, algorithm)

    if args.local_rank in [-1, 0]:
        logging.info(
            f"Loaded best validation model from epoch {selected_epoch}. "
            f"Now running final test on fusion branch only."
        )

    for test_env_name, test_env_loader in test_loaders.items():
        test_metrics, _ = algorithm_validate(
            algorithm,
            test_env_loader,
            writer,
            selected_epoch,
            f'final_test_{test_env_name}',
            branch=SELECTOR_BRANCH,
        )

        if args.local_rank in [-1, 0]:
            logging.info(
                f"[FINAL TEST][{test_env_name}][fusion] "
                f"Acc={test_metrics['acc']:.6f}, "
                f"MacroF1={test_metrics['macro_f1']:.6f}, "
                f"WeightedOVR-AUC={test_metrics['weighted_ovr_auc']:.6f}"
            )

    if args.local_rank in [-1, 0]:
        save_checkpoint(
            final_ckpt_path,
            algorithm,
            optimizer,
            scheduler,
            cfg.EPOCHS,
            best_performance,
            best_selector_score=best_selector_score,
            best_selector_epoch=best_selector_epoch,
            best_selector_metrics=best_selector_metrics,
        )
        with open(os.path.join(log_path, 'done'), 'w') as f:
            f.write(
                f'done, '
                f'best_selector_epoch={best_selector_epoch}, '
                f'best_selector_score={best_selector_score}, '
                f'best_selector_metrics={best_selector_metrics}'
            )
    if writer:
        writer.close()
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
