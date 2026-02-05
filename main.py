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

    state = {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 'best_performance': best_performance}
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
    scheduler.load_state_dict(checkpoint['scheduler_state'])

    start_epoch = checkpoint['epoch'] + 1
    best_performance = checkpoint.get('best_performance', 0.0)

    return start_epoch, best_performance


def main():
    start_time = time.time()
    args = get_args()

    args.output = os.path.abspath(args.output)

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

    cfg = setup_cfg(args)
    log_path = os.path.join(args.output, cfg.OUTPUT_PATH)

    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    writer = None
    if args.local_rank in [-1, 0]:
        writer = init_log(args, cfg, log_path, 0, [0, 0, 0])
        logging.info(f"Distributed: {is_distributed}, Rank: {args.local_rank}")
        print(f"[INFO] Log Path (Absolute): {log_path}")

    set_seed(args.seed)

    latest_ckpt_path = os.path.join(log_path, 'latest_model.pth')
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
    algorithm.network = algorithm.network.to(args.device)

    if hasattr(algorithm, 'classifier'):
        algorithm.classifier = algorithm.classifier.to(args.device)
    if hasattr(algorithm, 'swad_algorithm'):
        algorithm.swad_algorithm = algorithm.swad_algorithm.to(args.device)

    if is_distributed:
        algorithm.network = nn.SyncBatchNorm.convert_sync_batchnorm(algorithm.network)
        debug_log("SyncBatchNorm Activated.", args.local_rank)

        algorithm.network = DDP(algorithm.network, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = algorithm.optimizer
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

    iterator = tqdm(range(start_epoch - 1, cfg.EPOCHS), disable=(args.local_rank not in [-1, 0]), initial=start_epoch - 1, total=cfg.EPOCHS)

    for i in iterator:
        epoch = i + 1

        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(i)

        loss_avg = LossCounter()
        algorithm.train()

        for image, mask, label, domain, img_index in train_loader:
            image = image.to(args.device)
            mask = mask.to(args.device)
            label = label.to(args.device).long()
            domain = domain.to(args.device).long()

            minibatch = [image, mask, label, domain]
            loss_dict_iter = algorithm.update(minibatch)
            loss_avg.update(loss_dict_iter['loss'])

        if hasattr(algorithm, 'update_epoch'):
            algorithm.update_epoch(epoch)

        if args.local_rank in [-1, 0]:
            update_writer(writer, epoch, scheduler, loss_avg)

        scheduler.step()

        if args.local_rank in [-1, 0]:
            save_checkpoint(latest_ckpt_path, algorithm, optimizer, scheduler, epoch, best_performance)

        if is_distributed:
            dist.barrier()

        if epoch % cfg.VAL_EPOCH == 0:
            if args.local_rank in [-1, 0]:
                logging.info(f"Epoch {epoch} Validation...")

            val_auc, _ = algorithm.validate(val_loader, test_loader, writer)

            if val_auc > best_performance:
                best_performance = val_auc
                if args.local_rank in [-1, 0]:
                    logging.info(f"New Best Model! Val AUC: {val_auc:.4f}")
                    algorithm.save_model(log_path)

            if is_distributed:
                dist.barrier()

        if args.time_limit > 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > (args.time_limit - 300):
                debug_log(f"⏰ Time limit reached. Saving latest and exiting cleanly (Code 0).", args.local_rank)

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

    algorithm.renew_model(log_path)

    _, test_auc = algorithm.validate(val_loader, test_loader, writer)

    if args.local_rank in [-1, 0]:
        logging.info(f"Final Test AUC: {test_auc:.4f}")
        with open(os.path.join(log_path, 'done'), 'w') as f:
            f.write('done')
        if writer: writer.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()