# -*- coding: utf-8 -*-
# @Author: yuan
# @Date:   2020-12-24 18:04:09
# @Last Modified by:   yuan
# @Last Modified time: 2020-12-25 17:58:03
import torch
import os
import cv2
import time
import datetime
import csv
import collections
import copy
import shutil
import numpy as np
from detectron2.structures import Boxes, Instances
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from predictor import DefaultPredictor
from utils.parameter import default_argument_parser
from utils.loader import ModelLoader, DataLoader
from utils.visualizer import Visualize
from utils.allocate_gpu import occumpy_mem

from algorithm.auto_labelme import AutoLabelMe
import multiprocessing as mp

from collections import namedtuple
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlite_db import make_model  # CREATE TABLE


Detection = make_model('t_detection')


engine = create_engine('sqlite:///detection.db', echo=False)
Session = sessionmaker(bind=engine)


class ModelsPredict(ModelLoader):
    def __init__(self, result_folder, model_info, input_folder_path, **kwargs):
        self.result_folder = result_folder
        self.model_info = model_info
        self.input_folder_path = input_folder_path

        self.is_labelme = kwargs.get('auto_labelme', False)
        self.pool_size = kwargs.get('pool_size', 1)
        self.draw_top1 = kwargs.get('draw_top1', False)
        self.draw_total = kwargs.get('draw_total', False)
        self.write_style = kwargs.get('write_style', False)
        self.top1 = kwargs.get('top1', False)
        self.total = kwargs.get('total', False)
        self.show_masks = True

        self.image_size = tuple(self.model_info['image_size'])
        self.batch_size = self.model_info['batch_size']
        self.model_dicts = self.model_info['models']
        self.gid = kwargs.get('gid', None)
        self.coco_metadata_list = self.register_datasets()
        self.prediction_list = self.set_cfg_predictors()
        self.auto_labelme = AutoLabelMe()

    def register_datasets(self):
        coco_metadata_list = []
        for idx, model in enumerate(self.model_dicts):
            coco_metadata = super().register_dataset(os.path.split(
                self.input_folder_path)[1]+str(idx))
            print("coco_metadata", coco_metadata)
            coco_metadata_list.append(coco_metadata)
        return coco_metadata_list

    def set_cfg_predictors(self):
        # Multi-model for a gpu
        predictions = []
        for model in self.model_dicts:
            cfg = super().set_cfg(model)
            pred = DefaultPredictor(cfg, self.gid)
            predictions.append(pred)
            print("load", model)
        return predictions

    @staticmethod
    def create_instance(image_size, score, labels, bbox, masks):
        ret = Instances(image_size)
        ret.scores = torch.tensor(score)
        ret.pred_classes = torch.tensor(labels)
        ret.pred_masks = torch.tensor(masks)
        ret.pred_boxes = Boxes(torch.tensor(bbox))
        return ret

    def draw_img(self, file_name, img, instances,
                 coco_metadata, thing_classes):
        if self.draw_top1:
            result_file = os.path.join(self.result_folder, 'top1', file_name)

            if len(instances) == 0:
                self.visual_image_save(img, instances, coco_metadata,
                                       result_file,     thing_classes)
            else:
                self.visual_image_save(img, instances[0], coco_metadata,
                                       result_file, thing_classes)

        if self.draw_total:
            result_file = os.path.join(self.result_folder, 'total', file_name)
            self.visual_image_save(img, instances, coco_metadata, result_file,
                                   thing_classes)

    @staticmethod
    def visual_image_save(
            inputs, instance, coco_data_metadata, result_file,
            thing_classes, is_resize=False, resize_size=(128, 128)):

        v = Visualize(inputs[:, :, ::-1], metadata=coco_data_metadata)
        v = v.draw_instance_predictions(instance, thing_classes)

        img = v.get_image()[:, :, ::-1]
        if is_resize:
            img = cv2.resize(img, resize_size,
                             interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(result_file, img)

    @staticmethod
    def get_batch_instances(image_list, predictor):

        input_list = [cv2.imread(f) for f in image_list]
        batch_output = predictor(
            input_list)  # batch predict , gpu bounded
        batch_instances = [x['instances'].to('cpu') for x in batch_output]
        return input_list, batch_instances

    def save_csv_data(self, csv_top1_data, csv_total_data):
        if self.top1:
            csv_top1_file = open(os.path.join(
                self.result_folder, "top1.csv"), "a+")
            csv_top1_writer = csv.writer(csv_top1_file)
            for r in csv_top1_data:
                csv_top1_writer.writerow(r)
            csv_top1_file.close()
        if self.total:
            csv_total_file = open(os.path.join(
                self.result_folder, "total.csv"), "a+")
            csv_total_writer = csv.writer(csv_total_file)
            for r in csv_total_data:
                csv_total_writer.writerow(r)
            csv_total_file.close()

    def write_db(self, db_data_list: list):
        data_list = copy.deepcopy(db_data_list)
        for data in data_list:

            session = Session()
            detector = Detection()
            detector.model_name = data.model_name
            detector.product_name = data.product_name
            detector.site_name = data.site_name
            detector.lot_number = data.lot_number
            detector.serial_number = data.serial_number
            detector.process_time = datetime.datetime.now()
            detector.image_name = data.image_name
            detector.detection_path = data.detection_path
            detector.source_path = data.source_path
            detector.reference_path = data.reference_path
            detector.detection_class = data.detection_class  # top 1
            detector.true_label = data.true_label  # "無法得知"
            detector.confidence = data.confidence  # top 1
            detector.create_by = self.model_info['create_by']
            try:
                session.add(detector)
                session.commit()
            except Exception as e:
                print(e)
                session.rollback()
            finally:
                session.close()

    def filter_focus_label_scores(self, res: dict, layer: int):
        idx = 0
        scores = []
        labels = []
        bbox = []
        masks = []
        for _score, _class, label, mask, box, score in zip(
                res['scores'],
                res['pred_classes'],
                res['instances'].pred_classes,
                res['instances'].pred_masks,
                res['instances'].pred_boxes,
                res['instances'].scores):

            if (_class in self.model_dicts[layer]['focus_labels'].keys() and
                    _score >= self.model_dicts[layer]['focus_labels'][_class]):
                # print(_score, _class, label, mask, box, score)
                bbox.append(box.numpy())
                masks.append(mask.numpy())
                scores.append(float(score))
                labels.append(label)
            idx += 1
        return self.create_instance(self.image_size,
                                    scores, labels, bbox, masks)

    def get_batch_results(self,
                          layer: int, batch_instances, waiting_list: list,
                          input_list: list) -> dict:
        result = collections.defaultdict(dict)  # for each images
        for img_id, instances in enumerate(batch_instances):
            img_name = waiting_list[img_id]
            if self.show_masks is False and instances.has("pred_masks"):
                instances.remove('pred_masks')
            # find defect for this preidctor
            if len(instances) != 0:
                one_hot = [int(f) for f in instances.pred_classes]
                result[img_name]['scores'] = [
                    float(f) for f in instances.scores]
                result[img_name]['pred_classes'] = [
                    self.model_dicts[layer]['labels'][f] for f in one_hot]
            # not find defect for this preidctor
            else:
                result[img_name]['scores'] = [1.0]
                result[img_name]['pred_classes'] = ['OK']
            result[img_name]['instances'] = instances
            result[img_name]['image'] = input_list[img_id]
        return result

    def task(self, batch_filename: list):
        """
        self.prediction_list = [predictor],   
        if multi-layer model ,it looks like [predictor1, predcitor2]
        """
        csv_top1_data = []
        csv_total_data = []
        db_top1_list = []
        db_total_list = []
        db_data = namedtuple('db_data', ['model_name', 'product_name',
                                         'site_name', 'lot_number',
                                         'serial_number',
                                         'image_name',
                                         'source_path', 'reference_path',
                                         'detection_path', 'detection_class',
                                         'true_label', 'confidence'])

        waiting_list = copy.deepcopy(batch_filename)
        for layer, prediction in enumerate(self.prediction_list):

            if (len(waiting_list) == 0 and
                    len(csv_top1_data) == len(batch_filename)):
                break

            input_list, batch_instances = self.get_batch_instances(
                waiting_list, prediction)

            result = self.get_batch_results(
                layer, batch_instances, waiting_list, input_list)

            next_waiting_list = []
            for img_name, res in result.items():
                file_name = os.path.split(img_name)[1]
                # scores = res['scores']
                # pred_classes = res['pred_classes']

                ret = self.filter_focus_label_scores(
                    res, layer)

                if len(ret) == 0:
                    next_waiting_list.append(img_name)

                # save defect detection
                else:
                    print("file_name : {} is done~~~~".format(file_name))
                    self.draw_img(
                        file_name, res['image'], ret,
                        self.coco_metadata_list[layer],
                        self.model_dicts[layer]['labels'])
                    # get csv information
                    for j in range(len(res['pred_classes'])):
                        csv_data = tuple((file_name, res['pred_classes'][j],
                                          round(res['scores'][j], 5),
                                          datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                          'time per image', str(layer)))

                        _data = db_data(
                            model_name=self.model_dicts[layer]['model_name'],
                            product_name=self.model_info['product_name'],
                            site_name=self.model_info['site_name'],
                            lot_number='lot_number',
                            serial_number='serial_number',
                            # process_time=datetime.datetime.now(),
                            image_name=file_name,
                            source_path=img_name,
                            reference_path='reference_path',
                            detection_path=img_name,
                            detection_class=res['pred_classes'][j],
                            true_label=res['pred_classes'][j],
                            confidence=round(
                                res['scores'][j], 5),

                        )

                        if self.write_style == 'csv' and self.top1 and j == 0:
                            csv_top1_data.append(csv_data)
                        if self.write_style == 'csv' and self.total:
                            csv_total_data.append(csv_data)
                        if self.write_style == 'db' and self.top1 and j == 0:
                            db_top1_list.append(_data)
                        if self.write_style == 'db' and self.top1:
                            db_total_list.append(_data)

                    if self.is_labelme:
                        labelme_path = os.path.join(self.result_folder,
                                                    'labelme-' +
                                                    str(self.model_dicts[layer]
                                                        ['threshold']),
                                                    file_name)
                        os.makedirs(os.path.split(labelme_path)
                                    [0], exist_ok=True)
                        cv2.imwrite(labelme_path, res['image'])

                        self.auto_labelme.result2labelme(self.model_dicts[layer]["labels"], res['instances'],
                                                         os.path.split(
                            labelme_path)[0],
                            file_name,
                        )

            waiting_list = next_waiting_list

        # save ok detection
        if len(waiting_list) != 0:
            for img_name in waiting_list:
                file_name = os.path.split(img_name)[1]
                res = result[img_name]
                print("OUT OK file_name : {} is done~~~~".format(file_name))
                self.draw_img(file_name, res['image'], res['instances'],
                              self.coco_metadata_list[-1],
                              self.model_dicts[-1]['labels'])
                csv_data = tuple((os.path.split(img_name)[1], 'OK',  1.0,
                                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  'time per image', str(layer+1)))
                _data = db_data(
                    model_name=self.model_dicts[layer]['model_name'],
                    product_name=self.model_info['product_name'],
                    site_name=self.model_info['site_name'],
                    lot_number='lot_number',
                    serial_number='serial_number',
                    # process_time=datetime.datetime.now(),
                    image_name=file_name,
                    source_path=img_name,
                    reference_path='reference_path',
                    detection_path=img_name,
                    detection_class=res['pred_classes'][0],
                    true_label=res['pred_classes'][0],
                    confidence=res['scores'][0],
                )
                if self.write_style == 'csv' and self.top1:
                    csv_top1_data.append(csv_data)
                if self.write_style == 'csv' and self.total:
                    csv_total_data.append(csv_data)

                if self.is_labelme:
                    labelme_path = os.path.join(self.result_folder,
                                                'temp-' +
                                                str(self.model_dicts[-1]
                                                    ["threshold"]),
                                                file_name)
                    os.makedirs(os.path.split(labelme_path)[0], exist_ok=True)
                    cv2.imwrite(
                        labelme_path, res['image'])
                if self.write_style == 'db' and self.top1:
                    db_top1_list.append(_data)
                if self.write_style == 'db' and self.total:
                    db_total_list.append(_data)
        return csv_top1_data, csv_total_data, db_top1_list, db_total_list

    def run(self, filename_list: list):
        with ThreadPoolExecutor(max_workers=self.pool_size) as pool:
            # threads = []
            # for batch_filename in filename_list:
            #     t = pool.submit(self.task, batch_filename)
            #     threads.append(t)
            threads = [
                pool.submit(self.task, batch_filename)
                for batch_filename in filename_list]
            for th in as_completed(threads):
                csv_top1_data, csv_total_data, db_top1_list, db_total_list = th.result()

                if self.write_style == 'csv':
                    self.save_csv_data(csv_top1_data, csv_total_data)
                elif self.write_style == 'db':
                    self.write_db(db_top1_list)  # 0 : db_top1,1:db_total
            pool.shutdown()


class DistributedPredictorModel(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, gpu_list):
        self.gpu_list = gpu_list

    def distribute(self, args: tuple, filename_list: list, model_dicts: dict):
        """
        Distribute each gpu response what files

        Args:
            filename_list (list) : filename list 
            model_dict (dict) : config of model
        Returns:
            dist_filename_list (list(list)) : every gpu reponse batch files
            model_predictor_list (list(ModelsPredict)) : every gpu construct ModelsPredict
        """

        model_predictor_list = []
        assert len(self.gpu_list) == len(filename_list) == len(
            model_dicts),  ' Check number of gpu and number of model dicts'

        for gid, (_, model_dict) in zip(range(len(self.gpu_list)), model_dicts.items()):
            _model = ModelsPredict(
                input_folder_path=model_dict['input_folder_path'],
                result_folder=model_dict['result_path'],
                model_info=model_dict,
                auto_labelme=args.auto_labelme,
                pool_size=args.pool_size,
                top1=args.top1, ctotal=args.total,
                write_style=args.mode,
                draw_top1=args.draw_top1, draw_total=args.draw_total, gid=gid)
            model_predictor_list.append(_model)

        return model_predictor_list


class ModelLoaders(ModelLoader):
    def __init__(self, model_info_file):
        self.model_loader = ModelLoader(model_info_file=model_info_file)
        # print("model_loader", self.model_loader)
        self.model_info = self.model_loader.get_model_info()
        self.dataloader = []
        self.filename_list_by_product = []

    def get_model_info(self):
        return self.model_info

    def init_dataloader(self, **kwargs):
        for product_name, model in self.model_info.items():
            self.dataloader.append(DataLoader(
                input_folder_path=model['input_folder_path'],
                result_folder=model['result_path'],
                write_style=args.mode,
                draw_top1=kwargs.get('draw_top1', False),
                draw_total=kwargs.get('draw_total', False),
                top1=kwargs.get('top1', False),
                total=kwargs.get('total', False)
            )
            )

    def get_file_list(self):
        product_file_list = []
        for loader in self.dataloader:
            file_list, _ = loader.load_file_list('all')
            product_file_list.append(file_list)
        return product_file_list

    def get_batch_lists(self, file_list_by_product: list):

        for image_list, (k, v) in zip(file_list_by_product, self.model_info.items()):
            self.filename_list_by_product.append(
                ModelLoader.get_batch_list(image_list, bz=int(v['batch_size']))
            )
        return self.filename_list_by_product




if __name__ == '__main__':

    start = time.time()
    args = default_argument_parser()
    print("args", args)
    if args.mode == 'db':
        Detection.metadata.create_all(engine)  # 创建表结构

    print(args.model_info_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in args.gpus])
    occumpy_mem('0' , 0.2)

    modelloaders = ModelLoaders(args.model_info_file)
    dist = DistributedPredictorModel(gpu_list=args.gpus)

    modelloaders.init_dataloader(
        draw_top1=args.draw_top1, draw_total=args.draw_total,
        top1=args.top1, total=args.total, write_style=args.mode)
    file_list_by_product = modelloaders.get_file_list()
    filename_list_by_product = modelloaders.get_batch_lists(
        file_list_by_product)

    model_predictor_list = dist.distribute(
        args, filename_list_by_product, modelloaders.get_model_info())

    mp.set_start_method('spawn')
    mp_list = []
    try:

        for i in range(len(args.gpus)):
            p = mp.Process(target=model_predictor_list[i].run, args=(
                filename_list_by_product[i],))
            mp_list.append(p)
        for p in mp_list:
            p.start()
        for p in mp_list:
            p.join()
    except Exception as e:
        print(e)
