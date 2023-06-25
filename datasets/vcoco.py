# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from pathlib import Path
from PIL import Image
import cv2
import json
import detectron2
import numpy as np

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

from augmentation.simpleAug import simpleAug
from augmentation.middleAug import middleAug


class VCOCO(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries, aug_type):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = range(29)

        self.AugType = aug_type
        if self.AugType=='middle':
            # get config files of the target model and generate model
            model_folder = "COCO-InstanceSegmentation/"
            model_name = "mask_rcnn_X_101_32x8d_FPN_3x.yaml" 
            model_path = model_folder + model_name
            # initialize a config to store model from yaml file 
            cfg = get_cfg()
            # get hyperparameter from config file and recover default config
            cfg.merge_from_file(model_zoo.get_config_file(model_path))
            # Set determination threshold for positive and negative samples
            cfg.MODEL.ROI_HEADS_SCORE_THRESH_TEST = 0.5
            # load the pre-trained model(parameter weights) to the config
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
            # use the above config to generate a default predictor to detect image
            self.mask_predictor = DefaultPredictor(cfg)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_anno = self.annotations[idx]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # 这里进行自定义的图像数据增强，应当输入img、w、h和boxes，输出增强后的图像img、w、h和boxes，由于实际调用__getitem__(self, idx)函数是在训练过程中，此时seed已经设置好，不用担心复现问题
        if self.AugType=='none':
            pass
        elif self.AugType=='simple':
            img, w, h, boxes = simpleAug(img, w, h, boxes)
        elif self.AugType=='middle':
            img, w, h, boxes = middleAug(img, w, h, img_anno, boxes, self.mask_predictor)
        else:
            pass

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or \
                   (hoi['object_id'] != -1 and hoi['object_id'] not in kept_box_indices):
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    if hoi['object_id'] == -1:
                        obj_labels.append(torch.tensor(len(self._valid_obj_ids)))
                    else:
                        obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    if hoi['object_id'] == -1:
                        obj_box = torch.zeros((4,), dtype=torch.float32)
                    else:
                        obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
        else:
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx
            target['img_id'] = int(img_anno['file_name'].rstrip('.jpg').split('_')[2])

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        return img, target

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_vcoco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2014', root / 'annotations' / 'trainval_vcoco.json'),
        'val': (root / 'images' / 'val2014', root / 'annotations' / 'test_vcoco.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_vcoco.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = VCOCO(image_set, img_folder, anno_file, transforms=make_vcoco_transforms(image_set),
                    num_queries=args.num_queries,
                    aug_type=args.data_augmentation)
    if image_set == 'val':
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
