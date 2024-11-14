import cv2
import PIL
import numpy as np
import torch
import os
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO


import utility as su


def _convert_img_CV2PIL(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    return image

# https://docs.ultralytics.com/modes/predict/#working-with-results
class YoloV8:
    def __init__(self, checkpoint, device=0):
        self.model = YOLO(checkpoint)
        self.class_names = self.model.names
        self.class_idx_by_name = {value: key for key, value in self.class_names.items()}

        self.verbose = False
        self.conf = 0.2
        self.iou = 0.7
        self.imgsz = 640
        self.half = False
        self.device = device  # 0/1/cpu
        self.max_det = 100  # default 300
        # elf.classes = [] # filter classes todo


    def __call__(self, source):
        # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.BaseTensor
        return self.model(source, conf=self.conf, iou=self.iou,
                                  half=self.half, device=self.device,
                                  imgsz=self.imgsz, max_det=self.max_det)


    def _predicts2regions(self, pred):
        # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
        # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Masks
        # mask.data 非原图尺寸，https://github.com/ultralytics/ultralytics/issues/2272
        # predicts size: one image

        if pred.masks is None:
            return None
        regions = []
        boxes = pred.boxes.cpu().numpy()
        masks = pred.masks.cpu()
        for cls, conf, xyxy, poly, binary_mask in zip(boxes.cls, boxes.conf, boxes.xyxy, masks.xy, masks.data.numpy()):
            label = self.class_names[int(cls)]
            binary_mask = cv2.resize(binary_mask, (pred.orig_img.shape[1], pred.orig_img.shape[0]))
            regions.append(su.LabeledBoundingBox(bbox=xyxy.tolist(), label=label, score=conf, mask=binary_mask, polygon=poly))
        return regions

    def process(self, source):

        resutls = self.model(source)
        batched_regions=[]
        for pred in resutls:
            regions = self._predicts2regions(pred)
            batched_regions.append(regions)

        return batched_regions, resutls


