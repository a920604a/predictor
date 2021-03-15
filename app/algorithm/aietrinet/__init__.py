# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author         :  yuan
@Version        :
------------------------------------
@File           :  _init_.py
@Description    :
@CreateTime     :  2020/2/24 22:08
------------------------------------
@ModifyTime     :
"""

from .config import add_trinet_config
from .trident_backbone import (
    TridentBottleneckBlock,
    build_trident_resnet_backbone,
    make_trident_stage,
)
from .trident_rpn import TridentRPN
from .trident_rcnn import TridentRes5ROIHeads, TridentStandardROIHeads
