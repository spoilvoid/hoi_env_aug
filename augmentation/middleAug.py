import torch
import detectron2
import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum, unique
import json
import random
import math
import cv2
from PIL import Image

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

_valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)

def compute_mask_IOU(mask1, mask2, width, height):
    mask_and = cv2.bitwise_and(mask1, mask2)

    # 统计非0像素值的像素点数量
    area = cv2.countNonZero(mask_and)
    return area/(width*height) <= 0.1

def compute_shift_distance(box, width, height):
    left_limit = -box[0]
    right_limit = width - box[2]
    horizontal_shift = int(torch.clamp(torch.round(torch.randn(1) * (right_limit - left_limit) / 10 + (right_limit + left_limit) / 2), left_limit, right_limit))

    top_limit = -box[1]
    bottom_limit = height - box[3]
    vertical_shift = int(torch.clamp(torch.round(torch.randn(1) * (bottom_limit - top_limit) / 10 + (bottom_limit + top_limit) / 2), top_limit, bottom_limit))

    return horizontal_shift, vertical_shift

def get_pred_masks(image, width, height, category_ids, canAug, integrated_index_list, boxes, mask_predictor):
    bbox_mask_list = []
    for i, index in enumerate(integrated_index_list):
        bbox_image = image.crop(tuple(boxes[index]))

        outputs = mask_predictor(np.array(bbox_image))
    
        # masks是一个0、1的遮罩图像集，每一个对应一个物体检测框，我们取第一个mask（因为他的得分score最高）
        if(not outputs['instances'].pred_masks.shape[0]):
            image_binary_mask = np.zeros((height, width), dtype=np.uint8)
            canAug[i] = False
        else:
            mask_flag = False
            for mask, category_index, score in zip(outputs["instances"].pred_masks, outputs["instances"].pred_classes, outputs["instances"].scores):
                if score<=0.5 or _valid_obj_ids[category_index] != category_ids[index]:
                    continue
                else:
                    bbox_mask = mask.cpu().numpy().astype(np.uint8)
                    mask_flag = True
                    break
            if(mask_flag):
                # 将mask转化为一个2值图集，为true的部分赋255，为false的部分赋0，为接下来取出mask对应的部分图像做准备
                _, bbox_binary_mask = cv2.threshold(bbox_mask, 0, 255, cv2.THRESH_BINARY)
                #最终产生的mask与instance_image是bbox大小的RGB图像，需要转化成原图大小
                image_binary_mask = cv2.copyMakeBorder(bbox_binary_mask, boxes[index][1], height-boxes[index][3], boxes[index][0], width-boxes[index][2], cv2.BORDER_CONSTANT, value=0)
            else:
                image_binary_mask = np.zeros((height, width), dtype=np.uint8)
                canAug[i] = False
        bbox_mask_list.append(image_binary_mask)
    return bbox_mask_list

def move_objects(image, width, height, boxes, aug_masks, aug_box_indices, augChoice):
    image_fill_mask = np.zeros((height, width), dtype=np.uint8)
    for mask, index, choice in zip(aug_masks, aug_box_indices, augChoice.tolist()):
        if choice:
            # 计算instance mask在两个方向上移动的距离
            horizontal_shift, vertical_shift = compute_shift_distance(boxes[index], width, height)
            # 记录产生的新box，便于后续输出
            boxes[index][0] = boxes[index][0] + horizontal_shift
            boxes[index][1] = boxes[index][1] + vertical_shift
            boxes[index][2] = boxes[index][2] + horizontal_shift
            boxes[index][3] = boxes[index][3] + vertical_shift
            # 形成平移后的mask与instance image
            M = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
            shift_mask = cv2.warpAffine(mask, M, (width, height))
            shift_instance_image = cv2.warpAffine(cv2.bitwise_and(image, image, mask=mask), M, (width, height))
            # 记录需要修复的mask区域为将mask——shift_mask粘贴到image_fill_mask上
            image_fill_mask = cv2.bitwise_or(image_fill_mask, cv2.bitwise_and(cv2.bitwise_not(shift_mask), mask))
            # 扣去mask与shift_mask对应区域，然后将shift_instance_image粘贴到image上
            image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(cv2.bitwise_or(mask, shift_mask)))# 扣去指定区域
            image = cv2.bitwise_or(image, shift_instance_image)# 粘贴到image上
    return image, image_fill_mask


'''对于中层的数据增强，我们进行图像中的HOI作用距离变换，即将HOI中的object bounding box进行位置移动，并对图像进行修复'''
# 由于设计HOI标注的部分，需要额外将annotation输入

def middleAug(image, width, height, annotation, boxes, mask_predictor):
    # image: PIL Image
    # boxes: torch
    # annotation: dictionary
    # 有1/10的概率进行数据增强，此时产生的随机数为0，对应的isAug为true
    isAug = (torch.randint(0,10,(1,)).item() == 0)
    if isAug:
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)
        boxes = torch.as_tensor(boxes, dtype=torch.int32).tolist()
        category_ids = [obj['category_id'] for obj in annotation['annotations']]
        # 首先需要根据IOU判断哪些boxes可以被中层数据增强（通过annotation略过对无HOI的物box检测，因为认为可以被破坏而不影响语义）
        # 数据集中category_id为1表示人，这类bbox无法进行移动，但是仍应判断与其它物体的IOU
        sub_index_list = set()
        obj_index_list = set()
        # 所有可能发生碰撞的bbox记录在sub_index_list和obj_index_list
        for hoi in annotation['hoi_annotation']:
            if hoi['subject_id'] < len(boxes):
                sub_index_list.add(hoi['subject_id'])
            if hoi['object_id'] != -1 and hoi['object_id'] < len(boxes):
                obj_index_list.add(hoi['object_id'])
        # sub_index_list与obj_index_list和的集合
        integrated_index_list = sub_index_list | obj_index_list
        # 表示对应的integrated_index_list元素能否增强
        canAug = [False if obj in sub_index_list else True for i, obj in enumerate(integrated_index_list)]
        del sub_index_list
        del obj_index_list
        # 记录对应的integrated_index_list元素的bbox对应的mask
        bbox_mask_list = get_pred_masks(image, width, height, category_ids, canAug, integrated_index_list, boxes, mask_predictor)
        del category_ids
        # 进行IOU计算，剔除不可以进行增强的bbox
        for i, mask1 in enumerate(bbox_mask_list):
            for j, mask2 in enumerate(bbox_mask_list):
                if canAug[i]==False and canAug[j]==False:
                    continue
                elif j<=i:
                    continue
                elif not compute_mask_IOU(mask1, mask2, width, height):
                    canAug[i] = False
                    canAug[j] = False
                else:
                    pass
        # aug_masks记录可以被数据增强的mask，aug_box_indices记录可以被数据增强的bbox在boxes对应的下标
        aug_masks = [mask for i, mask in enumerate(bbox_mask_list) if canAug[i]]
        aug_box_indices = [index for i, index in enumerate(integrated_index_list) if canAug[i]]
        del canAug
        del bbox_mask_list
        del integrated_index_list
        
        if(len(aug_box_indices)):
            # 对于每一个增强选项均有概率参与数据增强，为了保证至少有一项参与数据增强，因此，在所有选项均为false时，平等地随机选择一项进行数据增强，同时我们希望有进行多项增强的可能性
            augChoice = torch.randint(0,10,(len(aug_box_indices),)) == torch.zeros([len(aug_box_indices)])
            if(not augChoice.sum()):
                augChoice[torch.randint(0,len(augChoice),(1,)).item()] = True
            # 我们选择每次删除原instance区域时，使用一个mask进行区域记录，在最后一次性修复整个mask对应区域
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image, image_fill_mask = move_objects(image, width, height,  boxes, aug_masks, aug_box_indices, augChoice)
            del aug_masks
            del aug_box_indices
            del augChoice
            image = cv2.inpaint(image, image_fill_mask, 5, cv2.INPAINT_TELEA)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes = torch.as_tensor(boxes, dtype=torch.int32).reshape(-1, 4)
    return image, width, height, boxes
