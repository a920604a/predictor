#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import cv2
import sys
import uuid

import imgviz
import numpy as np

import labelme

import albumentations as A

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

def augumentation(out_ann_file, label_file, class_name_to_id, data, image_id, image, masks, bboxes, class_labels, output_dir, noviz):
    ## 1. 图像增强方法
    transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.IAAPerspective(p=0.3),
            A.IAAAffine(p=.1),
            A.IAAPiecewiseAffine(p=0.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    ## 2. 执行图像增强
    transformed = transform(image=image, masks=masks, bboxes=bboxes, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_masks = transformed['masks']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']

    ## 3. 保存增强后的图像
    transformed_base = osp.splitext(osp.basename(label_file.imagePath))[0]
    out_transformed_file = osp.join(output_dir, "images", transformed_base + str(image_id) + ".jpg")
    imgviz.io.imsave(out_transformed_file, transformed_image)

    ## 4. 生成coco json images信息
    data["images"].append(
        dict(
            license=0,
            url=None,
            file_name=osp.relpath(out_transformed_file, osp.dirname(out_ann_file)),
            height=transformed_image.shape[0],
            width=transformed_image.shape[1],
            date_captured=None,
            id=image_id,
        )
    )

    masks = {}  # for area
    segmentations = collections.defaultdict(list) # for segmentation
    for mask, bbox, class_label in zip(transformed_masks, transformed_bboxes, transformed_class_labels):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        group_id = uuid.uuid1()
        instance = (class_label, group_id)
        for contour in contours:
            points = []
            for cont in contour:
                [points.append(float(i)) for i in list(cont.flatten())]
            segmentations[instance].append(points)
            polygon = np.asarray(points).reshape(-1,2)

            mask = labelme.utils.shape_to_mask(transformed_image.shape[:2], polygon, shape_type="polygon")

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

        ann_mask = np.asfortranarray(masks[instance].astype(np.uint8))
        ann_mask = pycocotools.mask.encode(ann_mask)
        ann_area = float(pycocotools.mask.area(ann_mask))

        data["annotations"].append(
            dict(
                id=len(data["annotations"]),
                image_id=image_id,
                category_id=class_name_to_id[class_label],
                segmentation=segmentations[instance],
                area=ann_area,
                bbox=list(bbox),
                iscrowd=0,
            )
        )

    if not noviz:
        viz_labels, viz_captions, viz_masks = zip(
            *[
                (class_name_to_id[cnm], cnm, msk)
                for (cnm, gid), msk in masks.items()
                if cnm in class_name_to_id
            ]
        )

        viz = imgviz.instances2rgb(
            image=transformed_image,
            labels=viz_labels,
            masks=viz_masks,
            captions=viz_captions,
            font_size=15,
            line_width=2,
        )

        out_viz_file = osp.join(
            output_dir, "visualization", transformed_base + str(image_id) + ".jpg"
        )

        imgviz.io.imsave(out_viz_file, viz)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    parser.add_argument("--augm_times", help="augmentation times")
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "images"))
    os.makedirs(osp.join(args.output_dir, "labels"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "visualization"))
    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    ## 1. coco json整体结构
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    ## 2. 生成coco json标签类别
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,)
        )

    ## 3. coco json输出文件
    out_ann_file = osp.join(args.output_dir, "trainval.json")
    ## 4. 所有labelme生成的json文件
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    ## 5. 循环处理每一个labelme文件
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        if args.augm_times:
            image_id = image_id * int(args.augm_times)

        ## 5.1 获取当前labelme文件
        label_file = labelme.LabelFile(filename=filename)

        ## 5.2 当前labelme文件所在的目录
        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "images", base + ".jpg")

        ## 5.3 储存当前图片文件到输出录
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        ## 5.4 生成coco json images信息
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        shapes = collections.defaultdict(list) ## for albumentations
        label_name_to_value = {"_background_": 0} ## for albumentations
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in sorted(label_file.shapes, key=lambda x: x["label"]):
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)

            ## for albumentations
            shapes[instance].append(shape)

            if label in label_name_to_value:
                label_value = label_name_to_value[label]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label] = label_value
            ## end

        segmentations = dict(segmentations)

        shapes = dict(shapes)  ## for albumentations

        class_bboxes = []  ## for albumentations
        class_masks = []  ## for albumentations
        class_labels = []  ## for albumentations
        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            class_labels.append(cls_name)  ## for albumentations

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()
            class_bboxes.append(bbox)  ## for albumentations

            lbl, _ = labelme.utils.shapes_to_label(img.shape, shapes[instance], label_name_to_value)
            labelme.utils.lblsave(osp.join(args.output_dir, "labels", str(image_id) + ".png"), lbl)
            lbl_mask = cv2.imread(osp.join(args.output_dir, "labels", str(image_id) + ".png"))
            class_masks.append(lbl_mask)

            ## 5.5 生成coco json annotations信息
            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not args.noviz:
            labels, captions, masks = zip(
                *[
                    (class_name_to_id[cnm], cnm, msk)
                    for (cnm, gid), msk in masks.items()
                    if cnm in class_name_to_id
                ]
            )
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                masks=masks,
                captions=captions,
                font_size=15,
                line_width=2,
            )
            out_viz_file = osp.join(args.output_dir, "visualization", base + ".jpg")
            imgviz.io.imsave(out_viz_file, viz)

        ## 6. 执行图像增强任务
        if args.augm_times:
            for i in range(int(args.augm_times) - 1):
                print("Augmenting dataset from:{}, id:{}".format(filename, i + 1))
                augumentation(out_ann_file=out_ann_file, label_file=label_file, class_name_to_id=class_name_to_id,
                    data=data, image_id=image_id + (i + 1), image=img, masks=class_masks, bboxes=class_bboxes, class_labels=class_labels,
                    output_dir=args.output_dir, noviz=args.noviz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    ## python labelme2coco.py images results --labels labels.txt --augm_times 20
    main()
