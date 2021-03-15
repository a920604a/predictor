# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author         :  yuan
@Version        :
------------------------------------
@File           :  auto_labelme.py
@Description    :
@CreateTime     :  2020/2/24 22:08
------------------------------------
@ModifyTime     :
"""

import base64
import json


# xml解析工具
import os
import re
import numpy as np
import cv2
from detectron2.utils.visualizer import GenericMask


class ToolHelper():
    # 从json文件中提取原始标定的信息
    def parse_json(self, path):
        with open(path)as f:
            json_data = json.load(f)
        return json_data

    # 对图片进行字符编码
    def img2str(self, img_name):
        with open(img_name, "rb")as f:
            base64_data = str(base64.b64encode(f.read()))
        match_pattern = re.compile(r'b\'(.*)\'')
        base64_data = match_pattern.match(base64_data).group(1)
        return base64_data

    # 保存图片结果
    def save_img(self, save_path, img):
        cv2.imwrite(save_path, img)

    # 保存json结果
    def save_json(self, file_name, save_folder, dic_info):
        with open(os.path.join(save_folder, file_name), 'w') as f:
            json.dump(dic_info, f, indent=2)


class AutoLabelMe(object):
    def __init__(self):
        pass

    def result2points(self, x1, y1, x2, y2):
        return [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]

    def result2shape(self, label, points):
        return {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }

    def result2json(self, shapes, imagePath, imageData, imageHeight, imageWidth):
        return {
            "version": "4.2.9",
            "flags": {},
            "shapes": shapes,
            "imagePath": imagePath,
            "imageData": imageData,
            "imageHeight": imageHeight,
            "imageWidth": imageWidth
        }

    def result2labelme(self, classes, x, filePath, fileName):
        toolhelper = ToolHelper()

        imagePath = fileName
        imageData = toolhelper.img2str(os.path.join(filePath, fileName))
        imageHeight = x.image_size[0]
        imageWidth = x.image_size[1]

        shapes = []
        if x.has("pred_masks"):
            masks = np.asarray(x.pred_masks)
            polygons = [GenericMask(mask, imageHeight, imageWidth).polygons for mask in masks]

            for clazz, polygon in zip(x.pred_classes.numpy(), polygons):
                label = classes[clazz]
                shape = self.result2shape(label, polygon[0].reshape(-1, 2).astype(float).tolist())
                shapes.append(shape)
        else:
            for clazz, box in zip(x.pred_classes.numpy(), x.pred_boxes.tensor.numpy()):
                label = classes[clazz]

                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                points = self.result2points(x1, y1, x2, y2)
                shape = self.result2shape(label, points)
                shapes.append(shape)

        json = self.result2json(shapes, imagePath, imageData,imageHeight, imageWidth)
        toolhelper.save_json(fileName.split('.')[0] + ".json", filePath, json)
