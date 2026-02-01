from yacs.config import CfgNode as CN

###########################
# Config definition
###########################
_C = CN()
# -----------------------------------------------------------------------------
# FPT+ Options
# -----------------------------------------------------------------------------
_C.MODEL.FPT = CN()
# 预训练的大模型路径 (LPM), 例如 'google/vit-base-patch16-224' 或本地路径
_C.MODEL.FPT.LPM_PATH = "/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/checkpoints/dinov3_vitb16"
# 高分辨率输入尺寸 (Frozen ViT Input)
_C.MODEL.FPT.HIGH_RES_SIZE = 512
# 低分辨率输入尺寸 (Side Network Input)
_C.MODEL.FPT.LOW_RES_SIZE = 224
# 侧边网络的层数
_C.MODEL.FPT.SIDE_LAYERS = 12
# 提取特征的层索引 (对应 ViT-Base 的层)
_C.MODEL.FPT.LAYERS_TO_EXTRACT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_C.OUT_DIR = "./output"
_C.SEED = 42
_C.USE_CUDA = True
_C.VERBOSE = True
_C.DROP_LAST = False
_C.DROP_OUT = 0.0
_C.LOG_STEP = 5

# --- 补充参数 ---
_C.ALGORITHM = "GDRNet"
_C.BACKBONE = "resnet50"
_C.RANDOM = False
_C.OVERRIDE = True
_C.VAL_EPOCH = 1  # <--- 新增：每 1 个 Epoch 验证一次
# ----------------

###########################
# Base
###########################
_C.EPOCHS = 100
_C.LEARNING_RATE = 1e-3
_C.BATCH_SIZE = 16
_C.WEIGHT_DECAY = 5e-4
_C.MOMENTUM = 0.9

###########################
# Model
###########################
_C.MODEL = CN()
_C.MODEL.NAME = ""

###########################
# Optimizer
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = ""

###########################
# Transforms
###########################
_C.TRANSFORM = CN()
_C.TRANSFORM.NAME = []
_C.TRANSFORM.AUGPROB = 0.5

# ColorJitter (brightness, contrast, saturation, hue)
_C.TRANSFORM.COLORJITTER_B = 1
_C.TRANSFORM.COLORJITTER_C = 1
_C.TRANSFORM.COLORJITTER_S = 1
_C.TRANSFORM.COLORJITTER_H = 0.05

###########################
# Dataset
###########################
_C.DATASET = CN()
_C.DATASET.ROOT = ""
_C.DATASET.NUM_CLASSES = 5 # 确保是 5
_C.DATASET.SOURCE_DOMAINS = ()
_C.DATASET.TARGET_DOMAINS = ()

###########################
# GDRNet
###########################
_C.GDRNET = CN()
_C.GDRNET.BETA = 0.5
_C.GDRNET.TEMPERATURE = 0.1
_C.GDRNET.SCALING_FACTOR = 4.

###########################
# Fishr
###########################
_C.FISHR = CN()
_C.FISHR.NUM_GROUPS = 0
_C.FISHR.EMA = 0.
_C.FISHR.PENALTY_ANNEAL_ITERS = 0
_C.FISHR.LAMBDA = 0.

###########################
# DRGen
###########################
_C.DRGEN = CN()
_C.DRGEN.N_CONVERGENCE = 0
_C.DRGEN.N_TOLERANCE = 0
_C.DRGEN.TOLERANCE_RATIO = 0.