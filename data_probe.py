import torch
from tqdm import tqdm

from dataset.data_manager import get_dataset
from utils.args import get_args, setup_cfg


def run_dataset_probe(train_loader, max_batches=51, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_bg_ratio = 0.0
    fg_rgb_sum = torch.zeros(3, device=device)
    fg_pixel_count = 0
    seen_batches = 0

    print('🚀 启动数据集分布探针...')
    for i, minibatch in enumerate(tqdm(train_loader, desc='Probing')):
        image, mask = minibatch[0].to(device), minibatch[1].to(device)

        mask_bin = (mask > 0).to(torch.float32)
        bg_pixels = (mask_bin == 0).sum(dim=[1, 2, 3]).float()
        total_pixels = mask_bin.shape[1] * mask_bin.shape[2] * mask_bin.shape[3]
        bg_ratio = (bg_pixels / total_pixels).mean().item()
        total_bg_ratio += bg_ratio

        for b in range(image.shape[0]):
            img = image[b]
            msk = mask_bin[b]
            valid_pixels = img[:, msk[0] > 0]
            if valid_pixels.shape[1] > 0:
                fg_rgb_sum += valid_pixels.sum(dim=1)
                fg_pixel_count += valid_pixels.shape[1]

        seen_batches += 1
        if seen_batches >= max_batches:
            break

    avg_bg_ratio = total_bg_ratio / max(seen_batches, 1)
    if fg_pixel_count > 0:
        avg_fg_rgb = (fg_rgb_sum / fg_pixel_count).detach().cpu().numpy()
    else:
        avg_fg_rgb = [0.0, 0.0, 0.0]

    print('\n' + '=' * 40)
    print('📊 探针分析报告:')
    print(f'1. 平均背景占比: {avg_bg_ratio:.2%}')
    print(f'2. 前景有效区域 RGB 均值: {avg_fg_rgb}')
    print('=' * 40)
    print('⚠️ 决策建议:')
    if avg_bg_ratio > 0.4:
        print('-> 背景占比过大！建议在 DINO 分支加入基于 Mask 的裁剪，或在送入 ViT 前用前景均值填充背景。')
    else:
        print('-> 背景占比可控，可优先尝试均值填充来消除黑边伪特征。')
    print('=' * 40)


if __name__ == '__main__':
    args = get_args()
    cfg = setup_cfg(args)
    if 'local_rank' not in vars(args):
        args.local_rank = -1
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, _, _, _, _ = get_dataset(args, cfg)
    run_dataset_probe(train_loader)
