import cv2
import numpy as np
import random
from PIL import Image


class SPMixAugmentation:
    def __init__(self, prob=0.5, alpha=1.0):
        self.prob = prob
        self.alpha = alpha

        try:
            self.saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
        except AttributeError:
            raise ImportError("❌ [SPMix Error] 无法创建 StaticSaliencyFineGrained 检测器。\n请确保安装了包含扩展模块的 OpenCV: pip install opencv-contrib-python")

    def get_saliency_map(self, image_np):
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

        success, saliency_map = self.saliency_detector.computeSaliency(image_np)

        if not success:
            raise RuntimeError("❌ [SPMix Critical Failure] OpenCV Saliency 计算失败！\n请检查输入图像是否过暗、过亮或损坏。程序已终止以防止错误的实验结论。")

        min_val, max_val = saliency_map.min(), saliency_map.max()
        if max_val == min_val:
            raise RuntimeError("❌ [SPMix Error] 生成了无效的 Saliency Map (全图数值相同)。")

        saliency_map = (saliency_map - min_val) / (max_val - min_val + 1e-8)

        return saliency_map

    def __call__(self, img1_pil, img2_pil):
        if random.random() > self.prob:
            return img1_pil, 1.0

        img1 = np.array(img1_pil)
        img2 = np.array(img2_pil)

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        sal2 = self.get_saliency_map(img2)

        h, w = img1.shape[:2]
        lam_param = np.random.beta(self.alpha, self.alpha)

        cut_rat = np.sqrt(1. - lam_param)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        if cut_w <= 0 or cut_h <= 0:
            return img1_pil, 1.0

        sal2_blur = cv2.GaussianBlur(sal2, (51, 51), 0)
        _, _, _, max_loc2 = cv2.minMaxLoc(sal2_blur)
        cx2, cy2 = max_loc2

        x1_src = np.clip(cx2 - cut_w // 2, 0, w)
        y1_src = np.clip(cy2 - cut_h // 2, 0, h)
        x2_src = np.clip(cx2 + cut_w // 2, 0, w)
        y2_src = np.clip(cy2 + cut_h // 2, 0, h)

        real_w = x2_src - x1_src
        real_h = y2_src - y1_src

        if real_w <= 0 or real_h <= 0:
            return img1_pil, 1.0

        cx1 = np.random.randint(w)
        cy1 = np.random.randint(h)

        x1_dst = int(np.clip(cx1 - real_w // 2, 0, w))
        y1_dst = int(np.clip(cy1 - real_h // 2, 0, h))

        x2_dst = x1_dst + real_w
        y2_dst = y1_dst + real_h

        if x2_dst > w:
            diff = x2_dst - w
            real_w -= diff
            x2_src -= diff
            x2_dst = w

        if y2_dst > h:
            diff = y2_dst - h
            real_h -= diff
            y2_src -= diff
            y2_dst = h

        if real_w <= 0 or real_h <= 0:
            return img1_pil, 1.0

        img_aug = img1.copy()

        img_aug[y1_dst:y2_dst, x1_dst:x2_dst] = img2[y1_src:y2_src, x1_src:x2_src]

        paste_area = real_w * real_h
        total_area = w * h
        lam_adjusted = 1.0 - (paste_area / total_area)

        return Image.fromarray(img_aug), lam_adjusted