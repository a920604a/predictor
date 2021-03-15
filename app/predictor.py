# -*- coding: utf-8 -*-
# @Author: yuan
# @Date:   2020-12-24 18:06:05
# @Last Modified by:   yuan
# @Last Modified time: 2020-12-25 17:17:36
import torch
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from concurrent.futures import ThreadPoolExecutor, as_completed

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg, gid: int):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model = self.model.to(torch.device('cuda:{}'.format(gid)))
        print('Default Prediction init!!!!!!', gid)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image_list: list):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """

        with torch.no_grad():
            _batch_img = []
            for im in original_image_list:
                if self.input_format == "RGB":
                    im = im[:, :, ::-1]
                height, width = im.shape[:2]
                image = self.aug.get_transform(im).apply_image(
                    im)  # resize 800,800,3 and  normalization
                image = torch.as_tensor(image.astype(
                    "float32").transpose(2, 0, 1))  # hwc -> chw

                inputs = {"image": image, "height": height, "width": width}

                _batch_img.append(inputs)
            predictions = self.model(_batch_img)

            return predictions
