from yacs.config import CfgNode as CN

###########################
# Config definition
###########################
_C = CN()

# -----------------------------------------------------------------------------
# 1. 核心路径与环境设置 (System)
# -----------------------------------------------------------------------------
_C.OUT_DIR = "/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/Standard_Pipeline/output_esdg_h100"
_C.USE_CUDA = True
_C.SEED = 42
_C.num_workers = 8
_C.VERBOSE = True

# -----------------------------------------------------------------------------
# 2. 模型与权重设置 (Model - 关键修改)
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# 是否使用预训练权重 (必须为 True，否则很难收敛)
_C.MODEL.PRETRAINED = True
# 权重文件的绝对路径 (请根据你实际上传的位置修改这里！)
_C.MODEL.PRETRAINED_PATH = "/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/checkpoints/resnet50-19c8e357.pth"

_C.ALGORITHM = "GDRNet"
_C.BACKBONE = "resnet50"
_C.DG_MODE = "ESDG"

# -----------------------------------------------------------------------------
# 3. 数据集设置 (Dataset)
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/GDR_Formatted_Data"
_C.DATASET.NUM_CLASSES = 5
_C.DATASET.SOURCE_DOMAINS = ["MESSIDOR"]
_C.DATASET.TARGET_DOMAINS = ["APTOS", "DDR", "DEEPDR", "FGADR", "IDRID", "RLDR", "EYEPACS"]

# -----------------------------------------------------------------------------
# 4. 训练超参数 (Training Hyperparameters)
# -----------------------------------------------------------------------------
_C.EPOCHS = 100
_C.LEARNING_RATE = 0.004
_C.BATCH_SIZE = 32
_C.WEIGHT_DECAY = 5e-4
_C.MOMENTUM = 0.9

_C.DROP_LAST = False
_C.DROP_OUT = 0.0
_C.LOG_STEP = 5
_C.VAL_EPOCH = 1

# -----------------------------------------------------------------------------
# 5. GDRNet 专属超参数
# -----------------------------------------------------------------------------
_C.GDRNET = CN()
_C.GDRNET.BETA = 0.5
_C.GDRNET.TEMPERATURE = 0.01
_C.GDRNET.SCALING_FACTOR = 4.0

# -----------------------------------------------------------------------------
# 6. 数据增强
# -----------------------------------------------------------------------------
_C.TRANSFORM = CN()
_C.TRANSFORM.NAME = []
_C.TRANSFORM.AUGPROB = 0.5
_C.TRANSFORM.COLORJITTER_B = 1
_C.TRANSFORM.COLORJITTER_C = 1
_C.TRANSFORM.COLORJITTER_S = 1
_C.TRANSFORM.COLORJITTER_H = 0.05

# -----------------------------------------------------------------------------
# 7. 占位符
# -----------------------------------------------------------------------------
_C.OPTIM = CN()
_C.OPTIM.NAME = ""
_C.RANDOM = False
_C.OVERRIDE = True