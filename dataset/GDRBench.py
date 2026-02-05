import os
import os.path as osp
from torch.utils.data.dataset import Dataset
from PIL import Image
# --- [新增] 引入 SPMix 依赖 ---
import random
from .spmix_aug import SPMixAugmentation


# -----------------------------

class GDRBench(Dataset):

    # [修改] 仅在末尾追加了 spmix_prob 参数，默认 0.5 以兼容旧代码
    def __init__(self, root, source_domains, target_domains, mode, trans_basic=None, trans_mask=None, trans_fundus=None,
                 spmix_prob=0.5):
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

        self.trans_basic = trans_basic
        self.trans_fundus = trans_fundus
        self.trans_mask = trans_mask

        # --- [新增] SPMix 初始化逻辑 (插入在读取数据之前) ---
        # 仅在训练模式且概率大于0时启用
        self.use_spmix = (mode == "train" and spmix_prob > 0)
        if self.use_spmix:
            print(f"✅ [GDRBench] SPMix Augmentation Enabled! (Prob={spmix_prob})")
            self.spmix = SPMixAugmentation(prob=spmix_prob, alpha=1.0)
        else:
            self.spmix = None
        # ------------------------------------------------

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
            print(f"   Sample Image: {self.data[0]}")
            if mode == 'train':
                print(f"   Sample Mask:  {self.masks[0]}")

    def _read_data(self, input_domains, split):
        # [保持原状] 你的逻辑完全没动
        for domain_idx, dname in enumerate(input_domains):
            files_to_try = []
            if split == "test":
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
        # [保持原状] 你的逻辑完全没动
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

                final_mask_path = None

                base_rel_path, original_ext = osp.splitext(impath_in_txt)

                search_exts = list(dict.fromkeys(['.png', original_ext] + mask_extensions))

                for ext in search_exts:
                    try_path = osp.join(self.mask_dir, base_rel_path + ext)
                    if osp.exists(try_path):
                        final_mask_path = try_path
                        break
                if final_mask_path is None:
                    final_mask_path = osp.join(self.mask_dir, base_rel_path + ".png")

                self.data.append(final_image_path)
                self.masks.append(final_mask_path)
                self.label.append(label)
                self.domain.append(domain_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            image_path = self.data[index]
            data = Image.open(image_path).convert("RGB")

            if self.mode == "train":
                mask_path = self.masks[index]
                if osp.exists(mask_path):
                    mask = Image.open(mask_path).convert("L")
                else:
                    mask = Image.new('L', data.size, 255)

            label = self.label[index]
            domain = self.domain[index]

            # --- [新增] SPMix 增强逻辑 (插入点) ---
            # 插入在读取完 label/domain 之后，transform 之前
            if self.use_spmix:
                try:
                    # 1. 随机找另一张图 (Source)
                    rand_idx = random.randint(0, len(self.data) - 1)
                    src_image_path = self.data[rand_idx]
                    src_label = self.label[rand_idx]

                    # 2. 读取 Source 图
                    src_data = Image.open(src_image_path).convert("RGB")

                    # 3. 执行 SPMix (data 是 Target, src_data 是 Source)
                    mixed_data, lam = self.spmix(data, src_data)
                    data = mixed_data  # 替换 data

                    # 4. 标签重写 (Hard Label Adaptation)
                    if lam < 0.5:
                        label = src_label

                except Exception as e:
                    # [严格报错] 遇到问题直接抛出，不掩盖
                    print(f"❌ [SPMix Error] Failed at index {index}, Path: {image_path}")
                    raise e
            # ------------------------------------

            if self.trans_basic is not None:
                data = self.trans_basic(data)

            if self.mode == "train" and self.trans_mask is not None:
                mask = self.trans_mask(mask)

            if self.mode == "train":
                return data, mask, label, domain, index
            else:
                return data, label, domain, index

        except Exception as e:
            print(f"Error loading index {index}, path: {self.data[index]}")
            raise e