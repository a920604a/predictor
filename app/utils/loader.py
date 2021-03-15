import os
import csv
import json
import math
from algorithm.trinet.config import add_trinet_config
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog


class DataLoader(object):
    csv_tuple_data = ("image name", "predict class",
                      "confidence", "detection time", "detection use time", "level")

    def __init__(self, input_folder_path: str, result_folder: str, **kwargs):
        self.input_folder_path = input_folder_path
        self.result_folder = result_folder
        self.write_style = kwargs.get('write_style', False)
        self.draw_top1 = kwargs.get('draw_top1', False)
        self.draw_total = kwargs.get('draw_total', False)
        self.top1 = kwargs.get('top1', False)
        self.total = kwargs.get('total', False)

        os.makedirs(self.result_folder, exist_ok=True)

        self.mk_draw_dir()
        self.top1_file = os.path.join(self.result_folder, "top1.csv")
        self.total_file = os.path.join(self.result_folder, "total.csv")

        if self.write_style == 'csv':
            self.check_csv_exist()
        elif self.write_style == 'db':
            pass

    def load_file_list(self, mode: str, load_num_images=100):
        image_list = []
        count = 0
        if mode == 'batch':
            for (root, _, files) in os.walk(self.input_folder_path):
                for f in files:
                    if f.endswith((".bmp", ".jpg", ".jpeg", ".png")):
                        image_list.append(os.path.join(root, f))
                        count += 1
                        if count >= load_num_images:
                            return image_list
            return image_list
        elif mode == 'all':
            new_folder_list = []
            for (root, _, files) in os.walk(self.input_folder_path):
                for f in files:
                    if f.endswith((".bmp", ".jpg", ".jpeg", '.png')):
                        if root not in new_folder_list:
                            new_folder_list.append(root)
                        image_list.append(os.path.join(root, f))
                        count += 1
            return image_list, new_folder_list

    def mk_draw_dir(self):
        if self.draw_top1:
            result_draw_top1_folder = os.path.join(self.result_folder, "top1")
            os.makedirs(result_draw_top1_folder, exist_ok=True)
        if self.draw_total:
            result_draw_total_folder = os.path.join(
                self.result_folder, "total")
            os.makedirs(result_draw_total_folder, exist_ok=True)

    def check_csv_exist(self):
        if self.top1:
            if not os.path.exists(self.top1_file):
                csv_top1_file = open(self.top1_file, 'w')
                csv_top1_writer = csv.writer(csv_top1_file)
                csv_top1_writer.writerow(self.csv_tuple_data)
            else:
                csv_top1_file = open(self.top1_file, 'a+')
                csv_top1_writer = csv.writer(csv_top1_file)
            csv_top1_file.close()
        if self.total:
            if not os.path.exists(self.total_file):
                csv_total_file = open(self.total_file, 'w')
                csv_total_writer = csv.writer(csv_total_file)
                csv_total_writer.writerow(self.csv_tuple_data)
            else:
                csv_total_file = open(self.total_file, 'a+')
                csv_total_writer = csv.writer(csv_total_file)
            csv_total_file.close()


class ModelLoader:
    config_dict = {
        "RtaNet-50": "/app/algorithm/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml",
        "RtaNet-101": "/app/algorithm/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml",
        "MiaskNet-50": "/app/algorithm/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "MaskNet-101": "/app/algorithm/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "MaskNet-X101": "/app/algorithm/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        "MaskNet-X152": "/app/algorithm/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml",
        "TriNet": "/app/algorithm/configs/COCO-Detection/tridentnet_fast_R_101_C4_3x.yaml"
    }

    def __init__(self, model_info_file: str):
        # print(kwargs)
        self.model_info_file = model_info_file

    def get_model_info(self):
        with open(self.model_info_file, 'r') as f:
            model_info = json.loads(f.read())
        return model_info

    def register_dataset(self, dataset_name: str):
        return MetadataCatalog.get(dataset_name if len(dataset_name) else "__unused")

    def set_cfg(self, model_info: dict):
        """
        Set neural network config by model_info

        Args:
            model_info (dict) : model config
        Returns:
            cfg 
        """

        cfg = get_cfg()

        if model_info['algo_name'] == "TriNet":
            add_trinet_config(cfg)
            cfg.merge_from_file(
                ModelLoader.config_dict[model_info['algo_name']])

        elif model_info['algo_name'] == "MskNet-X101":
            cfg.merge_from_file(
                ModelLoader.config_dict[model_info['algo_name']])

        cfg.merge_from_list(['MODEL.WEIGHTS', model_info['model_file']])
        # cfg.DATASETS.TEST = (dataset_name,)
        # cfg.OUTPUT_DIR = args.result_folder
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(model_info['labels'])
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = model_info['threshold']
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_info['threshold']
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = model_info['threshold']
        # if 'FCOS' in cfg.MODEL.keys():
        #     cfg.MODEL.FCOS.INFERENCE_TH_TEST = score_thresh
        # if 'MEInst' in cfg.MODEL.keys():
        #     cfg.MODEL.MEInst.INFERENCE_TH_TEST = score_thresh

        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        cfg.freeze()
        return cfg

    @staticmethod
    def get_batch_list(image_list: list, bz: int) -> list:
        assert isinstance(image_list, list), 'image list must be list '
        assert isinstance(bz, int), 'batch size must be Integer'
        filename_list = []
        total = len(image_list)
        for idx in range(math.ceil(total / bz)):  # batch predict
            if idx == math.ceil(total / bz) - 1:
                filename_list.append(image_list[idx * bz:])
            else:
                filename_list.append(image_list[idx * bz: (1 + idx) * bz])
        return filename_list
