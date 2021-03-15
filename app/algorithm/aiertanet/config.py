# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author         :  yuan
@Version        :
------------------------------------
@File           :  config.py
@Description    :
@CreateTime     :  2020/2/24 22:08
------------------------------------
@ModifyTime     :
"""

from detectron2.config import CfgNode as CN


def add_rtanet_config(cfg):
    """
    Add config for RtaNet.
    """
    _C = cfg

    # ---------------------------------------------------------------------------- #
    # RtaNet Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.RETINANET = CN()

    # This is the number of foreground classes.
    _C.MODEL.RETINANET.NUM_CLASSES = 80

    _C.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]

    # Convolutions to use in the cls and bbox tower
    # NOTE: this doesn't include the last conv for logits
    _C.MODEL.RETINANET.NUM_CONVS = 4

    # IoU overlap ratio [bg, fg] for labeling anchors.
    # Anchors with < bg are labeled negative (0)
    # Anchors  with >= bg and < fg are ignored (-1)
    # Anchors with >= fg are labeled positive (1)
    _C.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
    _C.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]

    # Prior prob for rare case (i.e. foreground) at the beginning of training.
    # This is used to set the bias for the logits layer of the classifier subnet.
    # This improves training stability in the case of heavy class imbalance.
    _C.MODEL.RETINANET.PRIOR_PROB = 0.01

    # Inference cls score threshold, only anchors with score > INFERENCE_TH are
    # considered for inference (to improve speed)
    _C.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
    _C.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
    _C.MODEL.RETINANET.NMS_THRESH_TEST = 0.5

    # Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
    _C.MODEL.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    # Loss parameters
    _C.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
    _C.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
    _C.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1
