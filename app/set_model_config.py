# -*- coding: utf-8 -*-
# @Author: yuan
# @Date:   2020-12-22 15:58:30
# @Last Modified by:   yuan
# @Last Modified time: 2020-12-23 16:21:42


import os
from detectron2.config import get_cfg
from algorithm.rtanet.config import add_rtanet_config
from algorithm.trinet.config import add_trinet_config
from algorithm.coco_data import register_algo_configs, register_model_instance


def setup(model: dict, mode: str):

    config_dict = register_algo_configs()
    cfg = get_cfg()

    if model['algorithm'] == "TriNet":
        add_trinet_config(cfg)
        cfg.MODEL.WEIGHTS = "algorithm/ImageNetPretrained/MSRA/R-101.pkl"
        cfg.merge_from_file(config_dict[model['algorithm']])

    else:
        cfg.merge_from_file(config_dict[model['algorithm']])
        if model['algorithm'] == "RtaNet-50" or model['algorithm'] == "RtaNet-101":
            add_rtanet_config(cfg)

        elif model['algorithm'] == "MskNet-50":
            cfg.MODEL.WEIGHTS = "algorithm/ImageNetPretrained/MSRA/model_final_fpn_50_3x.pkl"

        elif model['algorithm'] == "MaskNet-101":
            cfg.MODEL.WEIGHTS = "algorithm/ImageNetPretrained/MSRA/model_final_fpn_101_3x.pkl"

        elif model['algorithm'] == "MaskNet-X101":
            cfg.MODEL.WEIGHTS = "algorithm/ImageNetPretrained/MSRA/model_final_fpn_x101_3x.pkl"
        elif model.algorithm == "MaskNet-X152":
            cfg.MODEL.WEIGHTS = "algorithm/ImageNetPretrained/MSRA/model_final_x152.pkl"

    # if cfg.MODEL.RETINANET.NUM_CLASSES:
        #cfg.MODEL.RETINANET.NUM_CLASSES = len(model['defect_name_list'])

    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(model['batch_size'])  # 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
        model['defect_name_list'])  # detect classes

    if mode == 'train':
        cfg.DATASETS.TRAIN = (model['dataset_name'], )
        cfg.DATASETS.TEST = ()
        cfg.SOLVER.CHECKPOINT_PERIOD = int(model['CHECKPOINT_PERIOD'])
    else:
        cfg.DATASETS.TRAIN = ()
        cfg.DATASETS.TEST = (model['dataset_name'], )
        assert os.path.exists(os.path.join(
            model['output_path'], 'model_final.pth')), 'not find model {}'.format(os.path.join(
                model['output_path'], 'model_final.pth'))
        cfg.MODEL.WEIGHTS = os.path.join(
            model['output_path'], 'model_final.pth')
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(
            model['confidence_threshold'])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(
            model['confidence_threshold'])
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = float(
            model['confidence_threshold'])
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    # cfg.NUM_GPUS = 2

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = model['output_path']

    # cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.SOLVER.IMS_PER_BATCH = int(model['batch_size'])
    cfg.SOLVER.BASE_LR = float(model['learning_rate'])
    cfg.SOLVER.MAX_ITER = int(model['epoch'])
    # Save a checkpoint after every this number of iterations

    cfg.freeze()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg
