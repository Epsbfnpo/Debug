import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
import time
import logging
import datetime
import sys
from tqdm import tqdm

import algorithms
from utils.args import get_args, setup_cfg
from utils.misc import init_log, LossCounter, get_scheduler, update_writer
from utils.validate import algorithm_validate
from dataset.data_manager import get_dataset


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
    if hasattr(algorithm.network, 'module'):
        model_state = algorithm.network.module.state_dict()
    else:
        model_state = algorithm.network.state_dict()

    state = {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optimizer.state_dict(),
             'scheduler_state': scheduler.state_dict(), 'best_performance': best_performance}
    torch.save(state, path)


def load_checkpoint(path, algorithm, optimizer, scheduler):
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location='cpu')

    if hasattr(algorithm, '_load_net_state_dict'):
        algorithm._load_net_state_dict(checkpoint['model_state'])
    else:
        if hasattr(algorithm.network, 'module'):
            algorithm.network.module.load_state_dict(checkpoint['model_state'])
        else:
            algorithm.network.load_state_dict(checkpoint['model_state'])

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    try:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    except Exception as e:
        print(f"[Warning] Failed to load scheduler state: {e}")

    start_epoch = checkpoint['epoch'] + 1
    best_performance = checkpoint.get('best_performance', 0.0)

    return start_epoch, best_performance


def main():
    start_time = time.time()
    args = get_args()
    cfg = setup_cfg(args)

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))
        args.device = torch.device('cuda', args.local_rank)
        is_distributed = True
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_distributed = False

    log_path = os.path.abspath(cfg.OUT_DIR)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    writer = None
    if args.local_rank in [-1, 0]:
        writer = init_log(args, cfg, log_path, 0, [0, 0, 0])
        logging.info(f"Distributed: {is_distributed}, Rank: {args.local_rank}")
        print(f"[INFO] Log Path (Absolute): {log_path}")

    set_seed(cfg.SEED)

    latest_ckpt_path = os.path.join(log_path, 'latest_model.pth')
    best_ckpt_path = os.path.join(log_path, 'best_checkpoint.pth')
    final_ckpt_path = os.path.join(log_path, 'final_model.pth')

    if os.path.exists(final_ckpt_path):
        if args.local_rank in [-1, 0]:
            print(f"✅ Found {final_ckpt_path}. Training already completed. Exiting.")
        if is_distributed:
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)

    debug_log("Loading datasets...", args.local_rank)
    train_loader, val_loader, test_loader, dataset_size, train_sampler = get_dataset(args, cfg)

    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)

    # [核心修复] 一次性将算法内的所有组件（包括 EMA, Projector, Predictor 等）移到 GPU
    algorithm.to(args.device)

    # DDP 仍需单独处理 network
    if is_distributed:
        algorithm.network = nn.SyncBatchNorm.convert_sync_batchnorm(algorithm.network)
        debug_log("SyncBatchNorm Activated.", args.local_rank)
        algorithm.network = DDP(algorithm.network, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    optimizer = algorithm.optimizer

    if hasattr(algorithm, 'scheduler'):
        scheduler = algorithm.scheduler
    else:
        scheduler = get_scheduler(optimizer, cfg.EPOCHS)

    start_epoch = 1
    best_performance = 0.0

    if os.path.exists(latest_ckpt_path):
        debug_log(f"Found {latest_ckpt_path}. Resuming training...", args.local_rank)
        try:
            start_epoch, best_performance = load_checkpoint(latest_ckpt_path, algorithm, optimizer, scheduler)
            debug_log(f"Resumed from Epoch {start_epoch}.", args.local_rank)
        except Exception as e:
            debug_log(f"Error loading checkpoint: {e}. Starting from scratch.", args.local_rank)
    else:
        if args.local_rank == 0:
            print(f"\n[DEBUG] ❌ Checkpoint NOT found at: {latest_ckpt_path}")
        debug_log("No checkpoints found. Starting from scratch.", args.local_rank)

    iterator = tqdm(range(start_epoch - 1, cfg.EPOCHS), disable=(args.local_rank not in [-1, 0]),
                    initial=start_epoch - 1, total=cfg.EPOCHS)

    accum_iter = cfg.ACCUM_ITER

    for i in iterator:
        epoch = i + 1

        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(i)

        loss_avg = LossCounter()
        algorithm.train()

        # [修复点] 正确解包 9 个返回值
        for step, (img_weak, img_strong, img_mixed, mask, label1, label2, lam, domain, img_index) in enumerate(
                train_loader):
            img_weak = img_weak.to(args.device)
            img_strong = img_strong.to(args.device)
            img_mixed = img_mixed.to(args.device)
            mask = mask.to(args.device)
            label1 = label1.to(args.device).long()
            label2 = label2.to(args.device).long()
            lam = lam.to(args.device).float()
            domain = domain.to(args.device).long()

            # [修复点] 构造完整的 minibatch 列表，传给 algorithms.py
            minibatch = [img_weak, img_strong, img_mixed, mask, label1, label2, lam, domain, img_index]

            if cfg.ALGORITHM == 'GDRNet':
                loss_dict_iter = algorithm.update(minibatch, step=step, accum_iter=accum_iter)
            else:
                loss_dict_iter = algorithm.update(minibatch)

            loss_avg.update(loss_dict_iter)

        if hasattr(algorithm, 'update_epoch'):
            algorithm.update_epoch(epoch)

        if args.local_rank in [-1, 0]:
            update_writer(writer, epoch, scheduler, loss_avg)
            save_checkpoint(latest_ckpt_path, algorithm, optimizer, scheduler, epoch, best_performance)

        if is_distributed:
            dist.barrier()

        if epoch % cfg.VAL_EPOCH == 0:
            if args.local_rank in [-1, 0]:
                logging.info("Starting validation...")

            val_metrics, _ = algorithm.validate(val_loader, None, writer)
            val_auc = val_metrics['auc']

            if args.local_rank in [-1, 0]:
                logging.info(f"Epoch: {epoch} | Val AUC: {val_auc:.4f} | Test Skipped")

                if val_auc > best_performance:
                    best_performance = val_auc
                    save_checkpoint(best_ckpt_path, algorithm, optimizer, scheduler, epoch, best_performance)
                    algorithm.save_model(log_path)
                    logging.info(f"Best Val AUC: {val_auc:.4f} (Model Saved)")

            if is_distributed:
                dist.barrier()

        if hasattr(cfg, 'TIME_LIMIT') and cfg.TIME_LIMIT > 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > (cfg.TIME_LIMIT - 300):
                debug_log(f"⏰ Time limit reached. Saving latest and exiting cleanly.", args.local_rank)
                if args.local_rank in [-1, 0]:
                    save_checkpoint(latest_ckpt_path, algorithm, optimizer, scheduler, epoch, best_performance)
                    if writer: writer.close()
                if is_distributed:
                    dist.barrier()
                    dist.destroy_process_group()
                sys.exit(0)

    debug_log("All epochs finished.", args.local_rank)

    if args.local_rank in [-1, 0]:
        logging.info("Saving Final Checkpoint...")
        save_checkpoint(final_ckpt_path, algorithm, optimizer, scheduler, cfg.EPOCHS, best_performance)

    if is_distributed:
        dist.barrier()

    if args.local_rank in [-1, 0]:
        logging.info("Loading Best Model for Final Testing...")

    algorithm.renew_model(log_path)

    _, test_metrics = algorithm.validate(val_loader, test_loader, writer)

    if args.local_rank in [-1, 0]:
        logging.info("******************************************")
        logging.info(f"Final Test Performance (Best Model):")
        logging.info(f"AUC   : {test_metrics['auc']:.4f}")
        logging.info(f"ACC   : {test_metrics['acc']:.4f}")
        logging.info(f"F1    : {test_metrics['f1']:.4f}")
        logging.info(f"Kappa : {test_metrics['qwk']:.4f}")
        logging.info("******************************************")

        with open(os.path.join(log_path, 'done'), 'w') as f:
            f.write('done')
        if writer:
            writer.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()