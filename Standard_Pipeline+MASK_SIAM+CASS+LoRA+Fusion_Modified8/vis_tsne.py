import os
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from configs.defaults import get_cfg_defaults
from dataset.data_manager import get_dataset
from algorithms import get_algorithm_class


def extract_features(algorithm, data_loader, domain_name):
    """
    从指定的数据加载器中提取特征、标签和域信息
    """
    algorithm.eval()
    features_list = []
    labels_list = []
    domains_list = []

    with torch.no_grad():
        for image, mask, label, domain_idx in tqdm(data_loader, desc=f"Extracting {domain_name}"):
            image = image.cuda()
            # 在推理阶段，我们只使用 CNN 分支提取特征
            # 调用我们在 CASS_GDRNet 中定义的特征提取或直接 forward
            res = algorithm.network(x_cnn=image, x_vit=image)

            # 使用投影后的特征(proj_cnn)或分类前的特征(feat_cnn_final)
            # 这里推荐使用 proj_cnn，因为它是经过 CASS 和 FastMoCo 约束的纯净隐空间
            feat = res['proj_cnn'].cpu().numpy()

            features_list.append(feat)
            labels_list.append(label.numpy())
            domains_list.extend([domain_name] * image.size(0))

    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0), domains_list


def main():
    parser = argparse.ArgumentParser(description="t-SNE Visualization for GDRNet")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the saved model checkpoint (e.g., best_model.pth)")
    parser.add_argument("--output_dir", type=str, default="./figures", help="Directory to save t-SNE plots")
    parser.add_argument("--max_samples_per_domain", type=int, default=500,
                        help="Max samples per domain to avoid overcrowding")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载配置
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    # 2. 初始化模型并加载权重
    print("=> Building model...")
    algorithm_class = get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
    algorithm.cuda()

    print(f"=> Loading checkpoint from {args.ckpt}...")
    state_dict = torch.load(args.ckpt, map_location='cuda')
    # 兼容是否带有 module 前缀
    if hasattr(algorithm.network, 'module'):
        algorithm.network.module.load_state_dict(state_dict)
    else:
        try:
            algorithm.network.load_state_dict(state_dict)
        except:
            # 如果是 baseline (ERM)，它可能没有 network 这个层级，直接加载
            algorithm.load_state_dict(state_dict, strict=False)

    # 3. 遍历所有数据集提取特征
    all_domains = ['APTOS', 'DeepDR', 'FGADR', 'IDRID', 'Messidor', 'RLDR']

    all_features, all_labels, all_domain_names = [], [], []

    for domain_name in all_domains:
        # 这里为了可视化，临时把当前域设为 target，这样 get_dataset 就会返回它的 test_loader
        cfg.defrost()
        cfg.DATASET.TARGET_DOMAINS = [domain_name]
        cfg.freeze()

        _, _, test_loader = get_dataset(cfg)

        feats, labels, domain_names = extract_features(algorithm, test_loader, domain_name)

        # 随机采样，防止点太多导致 t-SNE 糊成一团
        if len(feats) > args.max_samples_per_domain:
            indices = np.random.choice(len(feats), args.max_samples_per_domain, replace=False)
            feats = feats[indices]
            labels = labels[indices]
            domain_names = [domain_names[i] for i in indices]

        all_features.append(feats)
        all_labels.append(labels)
        all_domain_names.extend(domain_names)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domain_names = np.array(all_domain_names)

    print(f"=> Extracted total {len(all_features)} features. Running t-SNE...")

    # 4. 运行 t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(all_features)

    # 5. 绘制顶会风格的对比图 (双子图)
    print("=> Plotting...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：按 Domain 着色 (期待看到各颜色完美混合)
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=all_domain_names,
        palette="Set1",
        s=40, alpha=0.8,
        ax=axes[0]
    )
    axes[0].set_title("Feature Space Colored by Domain (Hospital/Camera)", fontsize=14, fontweight='bold')
    axes[0].legend(title="Domain", loc='best')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # 右图：按 DR 等级 (Label) 着色 (期待看到同类聚集，不同类分离)
    # 定义类别名称
    class_names = ['Grade 0: Normal', 'Grade 1: Mild', 'Grade 2: Moderate', 'Grade 3: Severe', 'Grade 4: Proliferative']
    label_names = [class_names[l] for l in all_labels]

    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=label_names,
        hue_order=class_names,
        palette="viridis",  # 医学常用色系
        s=40, alpha=0.8,
        ax=axes[1]
    )
    axes[1].set_title("Feature Space Colored by DR Severity Grade", fontsize=14, fontweight='bold')
    axes[1].legend(title="DR Grade", loc='best')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "tsne_visualization.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"=> Saved t-SNE plots to {save_path}")


if __name__ == "__main__":
    main()