from detectron2.config import CfgNode as CN


def add_mobileinst_config(cfg):
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.MASK_ON = True
    # [MobileInst]
    cfg.MODEL.MOBILEINST = CN()

    # parameters for inference
    cfg.MODEL.MOBILEINST.CLS_THRESHOLD = 0.005
    cfg.MODEL.MOBILEINST.MASK_THRESHOLD = 0.45
    cfg.MODEL.MOBILEINST.MAX_DETECTIONS = 100

    # [Backbone]
    cfg.MODEL.TOPFORMER = CN()
    cfg.MODEL.TOPFORMER.NAME = "topformer_base"
    cfg.MODEL.TOPFORMER.OUT_FEATURES = ["x3", "x4", "x5", "x6"]
    cfg.MODEL.TOPFORMER.INJECTION = False
    cfg.MODEL.TOPFORMER.INJECTION_TYPE = "muli_sum"
    cfg.MODEL.TOPFORMER.NORM = 'SyncBN'

    cfg.MODEL.MOBILEINST.NORM = "SyncBN"

    # [Encoder]
    cfg.MODEL.MOBILEINST.ENCODER = CN()
    cfg.MODEL.MOBILEINST.ENCODER.NAME = "SEMaskEncoder"
    cfg.MODEL.MOBILEINST.ENCODER.IN_FEATURES = ["x3", "x4", "x5", "x6"]
    cfg.MODEL.MOBILEINST.ENCODER.NUM_CHANNELS = 256

    # [Decoder]
    cfg.MODEL.MOBILEINST.DECODER = CN()
    cfg.MODEL.MOBILEINST.DECODER.NAME = "MobileInstDecoder"
    cfg.MODEL.MOBILEINST.DECODER.NUM_MASKS = 100
    cfg.MODEL.MOBILEINST.DECODER.NUM_CLASSES = 20
    # kernels for mask features
    cfg.MODEL.MOBILEINST.DECODER.KERNEL_DIM = 128
    # upsample factor for output masks
    cfg.MODEL.MOBILEINST.DECODER.SCALE_FACTOR = 2.0
    # decoder.inst_branch
    cfg.MODEL.MOBILEINST.DECODER.KEY_DIM = 16
    cfg.MODEL.MOBILEINST.DECODER.NUM_HEADS = 8
    cfg.MODEL.MOBILEINST.DECODER.MLP_RATIO = 2
    cfg.MODEL.MOBILEINST.DECODER.ATTN_RATIO = 2

    # [Loss]
    cfg.MODEL.MOBILEINST.LOSS = CN()
    cfg.MODEL.MOBILEINST.LOSS.NAME = "MobileInstCriterion"
    cfg.MODEL.MOBILEINST.LOSS.ITEMS = ("labels", "masks")
    # loss weight
    cfg.MODEL.MOBILEINST.LOSS.CLASS_WEIGHT = 2.0
    cfg.MODEL.MOBILEINST.LOSS.MASK_PIXEL_WEIGHT = 5.0
    cfg.MODEL.MOBILEINST.LOSS.MASK_DICE_WEIGHT = 2.0
    # iou-aware objectness loss weight
    cfg.MODEL.MOBILEINST.LOSS.OBJECTNESS_WEIGHT = 1.0

    # [Matcher]
    cfg.MODEL.MOBILEINST.MATCHER = CN()
    cfg.MODEL.MOBILEINST.MATCHER.NAME = "MobileInstMatcher"
    cfg.MODEL.MOBILEINST.MATCHER.ALPHA = 0.8
    cfg.MODEL.MOBILEINST.MATCHER.BETA = 0.2

    # [Optimizer]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.5
    cfg.SOLVER.AMSGRAD = False

    # [Dataset mapper]
    cfg.MODEL.MOBILEINST.DATASET_MAPPER = "MobileInstDatasetMapper"
