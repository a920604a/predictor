from detectron2.utils.visualizer import (
    Visualizer,
    ColorMode,
    VisImage,
    GenericMask,
    _create_text_labels,
)


class Visualize(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE)
        # self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        # if metadata is None:
        #     metadata = MetadataCatalog.get("__nonexist__")
        # self.metadata = metadata
        # self.output = VisImage(self.img, scale=scale)
        # self.cpu_device = torch.device("cpu")

        # # too small texts are useless, therefore clamp to 9
        # self._default_font_size = max(
        #     np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        # )
        # self._instance_mode = instance_mode

    def draw_instance_predictions(self, predictions, thing_classes):
        boxes = predictions.pred_boxes if predictions.has(
            "pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has(
            "pred_classes") else None
        labels = _create_text_labels(classes, scores, thing_classes)
        keypoints = predictions.pred_keypoints if predictions.has(
            "pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height,
                                 self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
