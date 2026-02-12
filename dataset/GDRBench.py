import os
import os.path as osp
import torch
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from dataset.spmix_aug import SPMixAugmentation


class GDRBench(Dataset):
    def __init__(self, root, source_domains, target_domains, mode, trans_basic=None, trans_mask=None,
                 trans_fundus=None):
        if root is None:
            raise ValueError("Dataset root is None! Please check your config or args.")

        root = osp.abspath(osp.expanduser(root))
        self.mode = mode

        self.dataset_dir = osp.join(root, "images")
        self.mask_dir = osp.join(root, "masks")
        self.split_dir = osp.join(root, "splits")

        print(f"[{mode.upper()}] Dataset Root: {root}")
        print(f"[{mode.upper()}] Loading splits from: {self.split_dir}")

        self.data = []
        self.label = []
        self.domain = []
        self.masks = []

        # trans_basic: 通常是 Resize + Normalize + ToTensor (Weak Aug)
        # trans_fundus: 通常是 ColorJitter/Blur 等强增强 (Strong Aug)
        self.trans_basic = trans_basic
        self.trans_fundus = trans_fundus
        self.trans_mask = trans_mask

        # 初始化 SPMix，仅在训练模式启用
        self.spmix = SPMixAugmentation(prob=0.5, alpha=1.0) if mode == 'train' else None

        if mode == "train":
            self._read_data(source_domains, "train")
        elif mode == "val":
            self._read_data(source_domains, "crossval")
        elif mode == "test":
            self._read_data(target_domains, "test")

        if len(self.data) == 0:
            print(f"❌ [ERROR] No images loaded for mode '{mode}'!")
            raise RuntimeError(f"Found 0 images for {mode} set.")
        else:
            print(f"✅ [{mode.upper()}] Successfully loaded {len(self.data)} images.")

    def _read_data(self, input_domains, split):
        for domain_idx, dname in enumerate(input_domains):
            files_to_try = []
            if split == "test":
                # 测试集可能包含多个划分文件
                files_to_try.append(osp.join(self.split_dir, dname + "_test.txt"))
                files_to_try.append(osp.join(self.split_dir, dname + "_train.txt"))
                files_to_try.append(osp.join(self.split_dir, dname + "_crossval.txt"))
            else:
                files_to_try.append(osp.join(self.split_dir, dname + "_" + split + ".txt"))

            for file in files_to_try:
                if osp.exists(file):
                    self._read_split(file, domain_idx)
                elif split != "test":
                    print(f"⚠️ [WARNING] Split file not found: {file}")

    def _read_split(self, split_file, domain_idx):
        mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 2: continue

                impath_in_txt = parts[0]
                label = int(parts[1])
                final_image_path = osp.join(self.dataset_dir, impath_in_txt)

                # 尝试寻找对应的 Mask 文件
                final_mask_path = None
                base_rel_path, original_ext = osp.splitext(impath_in_txt)
                search_exts = list(dict.fromkeys(['.png', original_ext] + mask_extensions))
                for ext in search_exts:
                    try_path = osp.join(self.mask_dir, base_rel_path + ext)
                    if osp.exists(try_path):
                        final_mask_path = try_path
                        break
                if final_mask_path is None:
                    # 默认回退
                    final_mask_path = osp.join(self.mask_dir, base_rel_path + ".png")

                self.data.append(final_image_path)
                self.masks.append(final_mask_path)
                self.label.append(label)
                self.domain.append(domain_idx)

    def __len__(self):
        return len(self.data)

    def _load_image(self, index):
        """辅助函数：根据索引加载图像和标签"""
        path = self.data[index]
        img = Image.open(path).convert("RGB")
        label = self.label[index]
        return img, label

    def __getitem__(self, index):
        try:
            # 1. 加载当前的主图像 (img1)
            img1, label1 = self._load_image(index)
            domain = self.domain[index]

            # 加载 Mask (如有)
            mask_path = self.masks[index]
            if self.mode == "train" and osp.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new('L', img1.size, 255)  # 默认全白 Mask

            # 2. 训练模式逻辑
            if self.mode == "train":
                # === A. MASK_SIAM 专用流 (保持身份一致性) ===
                # Weak View: 基础增强
                if self.trans_basic is not None:
                    img_weak = self.trans_basic(img1)
                else:
                    img_weak = img1  # 应该是 Tensor

                # Strong View: 强增强 (如 FundusAug)
                # 如果没有专门的 trans_fundus，则使用 basic (这是为了兼容性，建议确保传入 trans_fundus)
                if self.trans_fundus is not None:
                    img_strong = self.trans_fundus(img1)
                else:
                    # 如果没有强增强，就再次做一次基础增强（因为 random crop/flip 会有差异）
                    img_strong = self.trans_basic(img1) if self.trans_basic else img1

                # === B. SPMix 专用流 (混合身份) ===
                # 随机抽取另一张图片 img2
                rand_idx = random.randint(0, len(self.data) - 1)
                img2, label2 = self._load_image(rand_idx)

                # 执行 SPMix
                # 返回混合后的 PIL 图片和混合比例 lambda
                img_mixed_pil, lam = self.spmix(img1, img2)

                # 将混合图转为 Tensor
                if self.trans_basic is not None:
                    img_mixed = self.trans_basic(img_mixed_pil)
                else:
                    img_mixed = img_mixed_pil

                # 处理 Mask 变换
                if self.trans_mask is not None:
                    mask = self.trans_mask(mask)

                # 返回完整元组：
                # (Weak, Strong, Mixed, Mask, Label1, Label2, Lambda, Domain, Index)
                return img_weak, img_strong, img_mixed, mask, label1, label2, np.float32(lam), domain, index

            # 3. 验证/测试模式逻辑
            else:
                if self.trans_basic is not None:
                    data = self.trans_basic(img1)
                else:
                    data = img1

                # [关键兜底修复] 确保 Mask 始终转为 Tensor
                if self.trans_mask is not None:
                    mask = self.trans_mask(mask)
                else:
                    # 如果 trans_mask 缺失，手动转为 Tensor 防止崩溃 (1, H, W)
                    mask_np = np.array(mask)
                    # 归一化到 [0, 1]
                    mask = torch.from_numpy(mask_np).float().unsqueeze(0) / 255.0

                # 验证模式下，Weak/Strong/Mixed 都是原图，Lambda=1.0
                return data, data, data, mask, label1, label1, 1.0, domain, index

        except Exception as e:
            print(f"Error loading index {index}, path: {self.data[index]}")
            # 遇到坏图可以尝试返回随机一张图，防止 dataloader 崩溃
            # 这里简单起见直接抛出异常
            raise e