# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author         :  yuan
@Version        :
------------------------------------
@File           :  change_msk.py
@Description    :
@CreateTime     :  2020/2/24 22:08
------------------------------------
@ModifyTime     :
"""

import torch
import numpy as np
import pickle


def changeMask(base_model, target_model, num_class):
    with open(base_model, 'rb') as f:
        # obj = f.read()
        weights = pickle.load(f, encoding='latin1')

    weights['model']['roi_heads.box_predictor.cls_score.weight'] = np.zeros(
        [num_class+1, 1024], dtype='float32')
    weights['model']['roi_heads.box_predictor.cls_score.bias'] = np.zeros(
        [num_class+1], dtype='float32')

    weights['model']['roi_heads.box_predictor.bbox_pred.weight'] = np.zeros(
        [num_class*4, 1024], dtype='float32')
    weights['model']['roi_heads.box_predictor.bbox_pred.bias'] = np.zeros(
        [num_class*4], dtype='float32')

    weights['model']['roi_heads.mask_head.predictor.weight'] = np.zeros(
        [num_class, 256, 1, 1], dtype='float32')
    weights['model']['roi_heads.mask_head.predictor.bias'] = np.zeros(
        [num_class], dtype='float32')

    with open(target_model, 'wb') as f:
        pickle.dump(weights, f)
