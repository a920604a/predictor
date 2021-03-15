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
import numpy as np
from detectron2.structures import Boxes, Instances
from concurrent.futures import ThreadPoolExecutor, as_completed
from predictor import DefaultPredictor
from utils.parameter import default_argument_parser
from utils.loader import ModelLoader, DataLoader
from utils.visualizer import Visualize
from utils.timer import time_wrapper
from utils.allocate_gpu import occumpy_mem


from algorithm.auto_labelme import AutoLabelMe
import multiprocessing as mp


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
                                       result_file, thing_classes)
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
    def get_batch_instances(image_list: list, predictor: DefaultPredictor):

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
                          layer: int, batch_instances: list, waiting_list: list,
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

    @time_wrapper
    def task(self, batch_filename: list, predictors: list, coco_metadata_list: list):
        """
        self.prediction_list = [predictor] , 
        if multi-layer model ,it looks like [predictor1, predcitor2]
        """
        csv_top1_data = []
        csv_total_data = []
        waiting_list = copy.deepcopy(batch_filename)
        # For model which has multi-predictor
        for layer, predictor in enumerate(predictors):

            if (len(waiting_list) == 0 and
                    len(csv_top1_data) == len(batch_filename)):
                break

            input_list, batch_instances = self.get_batch_instances(
                waiting_list, predictor)

            # put network's output on result (dict)
            result = self.get_batch_results(
                layer, batch_instances, waiting_list, input_list)
            # analysis and determine do what action
            next_waiting_list = []
            for img_name, res in result.items():
                file_name = os.path.split(img_name)[1]
                # scores = res['scores']
                # pred_classes = res['pred_classes']
                try:
                    ret = self.filter_focus_label_scores(
                        res, layer)
                except Exception as e:
                    print(e)
                if len(ret) == 0:
                    next_waiting_list.append(img_name)
                    print('file_name', file_name)

                # save defect detection
                else:
                    print("file_name : {} is done~~~~".format(file_name))
                    self.draw_img(
                        file_name, res['image'], ret,
                        coco_metadata_list[layer],
                        self.model_dicts[layer]['labels'])
                    # get csv information for total and top1
                    for j in range(len(res['pred_classes'])):
                        csv_data = tuple((file_name, res['pred_classes'][j],
                                          round(res['scores'][j], 5),
                                          datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                          'time per image', str(layer)))
                        if self.write_style == 'csv' and self.top1 and j == 0:
                            csv_top1_data.append(csv_data)
                        if self.write_style == 'csv' and self.total:
                            csv_total_data.append(csv_data)

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
                              coco_metadata_list[-1],
                              self.model_dicts[-1]['labels'])
                csv_data = tuple((os.path.split(img_name)[1], 'OK',  1.0,
                                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  'time per image', str(layer+1)))
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

        return csv_top1_data, csv_total_data

    def run(self, filename_list: list):
        # exec task and allocate multi-thread for one gpus
        # save csv results
        with ThreadPoolExecutor(max_workers=self.pool_size) as pool:
            # threads = []
            # for batch_filename in filename_list:
            #     t = pool.submit(self.task, batch_filename,
            #                     self.prediction_list, self.coco_metadata_list)
            #     threads.append(t)
            threads = [
                pool.submit(self.task, batch_filename,
                            self.prediction_list,
                            self.coco_metadata_list)
                for batch_filename in filename_list]

            for th in as_completed(threads):
                csv_top1_data, csv_total_data = th.result()
                if self.write_style == 'csv':
                    self.save_csv_data(csv_top1_data, csv_total_data)
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
        dist_filename_list = []
        batch = len(filename_list)//len(self.gpu_list)
        for gid in range(len(self.gpu_list)):
            _model = ModelsPredict(
                input_folder_path=args.input_folder_path,
                result_folder=args.result_folder,  model_info=model_dicts,
                auto_labelme=args.auto_labelme,
                pool_size=args.pool_size,
                top1=args.top1, total=args.total,
                write_style=args.mode,
                draw_top1=args.draw_top1, draw_total=args.draw_total, gid=gid)
            model_predictor_list.append(_model)

            if gid == len(self.gpu_list)-1:
                dist_filename_list.append(filename_list[batch*gid:])
            else:
                dist_filename_list.append(
                    filename_list[batch*gid:batch*(gid+1)])
        return dist_filename_list,  model_predictor_list


if __name__ == '__main__':
    occumpy_mem('0', 0.2)
    start = time.time()
    args = default_argument_parser()
    print("args", args)
    assert args.mode == 'csv', 'mode must be csv'

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in args.gpus])
    dataloader = DataLoader(
        input_folder_path=args.input_folder_path,
        result_folder=args.result_folder,
        write_style=args.mode,
        draw_top1=args.draw_top1, draw_total=args.draw_total,
        top1=args.top1, total=args.total)

    model_loader = ModelLoader(model_info_file=args.model_info_file)
    dist = DistributedPredictorModel(gpu_list=args.gpus)

    model_dicts = model_loader.get_model_info()

    file_list, _ = dataloader.load_file_list('all')

    filename_list = model_loader.get_batch_list(
        file_list, model_dicts['batch_size'])

    dist_filename_list,  model_list = dist.distribute(
        args, filename_list, model_dicts)

    print("load models spend {} sec".format((time.time() - start)))

    try:
        mp.set_start_method('spawn')
        mp_list = []
        for i in range(len(args.gpus)):
            p = mp.Process(target=model_list[i].run, args=(
                dist_filename_list[i], ))
            mp_list.append(p)
        for p in mp_list:
            p.start()
        for p in mp_list:
            p.join()
        print("{} image spends  {} sec".format(
            len(file_list), (time.time() - start)))
    except Exception as e:
        print(e)
