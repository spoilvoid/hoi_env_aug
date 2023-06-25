# import some common libraries
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

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.data import transforms as T


@unique
class TransformType(Enum):
    HFlipTransform = 1
    VFlipTransform = 2
    NoOpTransform = 3
    RandomCrop = 4
    RandomExtent = 5
    RandomRotation = 6
    RandomBrightness = 7
    RandomContrast = 8


@unique
class DatasetType(Enum):
    HICO_DET = 1
    V_COCO = 2


HICO_DET_CATEGORY_NUM = 80
HICO_DET_VERB_NUM = 117
HICO_DET_HOI_NUM = 600
V_COCO_CATEGORY_NUM = 81
V_COCO_VERB_NUM = 29
C_COCO_HOI_NUM = 0


# 特别地，对于V-COCO数据集，它存在一些没有对应物体的动作（verb），如微笑（smile）等，此时的object_id显示为-1，在处理时，我们将对应的obj_label设为-1，obj_box设为[0, 0, 0, 0],obj_center_points设为[0, 0]


# 输入标注所在的json文件的路径和所需要的标注（annotation）在数组中的下标（index）
# 输出对应的标注（annotation）条目，越界则输出False
def getAnno(json_path, index):
    with open(json_path,'r') as file:
        annotations = json.load(file)
    file.close()
    return annotations[index] if len(annotations) > index >= 0 else False


# 输入存放图片的文件夹的路径（image_folder_path）与该图片对应的标注（annotation）
# 输出读出的图片（image），图片形式为numpy格式，默认完整的数据集中标注存在则图片存在
def getImage(image_folder_path, annotation):
    image_path = os.path.join(image_folder_path, annotation['file_name'])
    image = plt.imread(image_path)
    return image


# 输入图像（image）、标注（annotation）、数据集类型标注（dataset_type）
# V-COCO数据集不含hoi_id、HICO-DET数据集含hoi_id
# 由于HICO-DET数据集中含有117个verb和600个hoi，且id从1开始计数，故范围为[1,118)和[1,601)，而V-COCO数据集中不涉及hoi类，仅29个verb类，无'hoi_category_id'，且id从0开始计数，故范围为[0,29)
# 可以根据输入的hoi_num是否为0判断数据集的类型
# 由于V-COCO的数据标注中不含图片的高与长，故最终输出的内容中不含图片的高与长，需另行获取,这里从图像（image）中获取
# 经过无用界限框（bbox）去除，合法检测等途径，最终输出包含所需信息的列表（target）
def analyseAnno(image, annotation, dataset_type):
    # boxes是标注中所有的界限框（bbox）
    boxes = torch.as_tensor([obj['bbox'] for obj in annotation['annotations']], dtype=torch.float32).reshape(-1, 4)
    # center_points是标注中所有界限框（bbox）的中心点（center_point）
    center_points = torch.mean(boxes.reshape(-1, 2, 2),1).reshape(-1, 2)
    # classes是标注中所有界限框（bbox）的类id（category_id）
    classes = torch.as_tensor([obj['category_id'] for obj in annotation['annotations']], dtype=torch.int64)
    height = image.shape[0]
    width = image.shape[1]
    # 防止center_points与boxes范围超出图像范围
    boxes[:, 0::2].clamp_(min=0, max=width)
    boxes[:, 1::2].clamp_(min=0, max=height)
    center_points[:, 0::2].clamp_(min=0, max=width)
    center_points[:, 1::2].clamp_(min=0, max=height)
    # 根据数据坐标比较，保留合法的box
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0]) 
    boxes = boxes[keep]
    center_points = center_points[keep]
    classes = classes[keep]
    # 将图片信息进行整合，然后放入target中形成json样式的数据记录
    target = {}
    target['width'] = torch.as_tensor(int(width))
    target['height'] = torch.as_tensor(int(height))
    target['boxes'] = boxes
    target['center_points'] = center_points 
    target['labels'] = classes

    # 对HOI相关部分进行过滤整合
    obj_labels, verb_labels, sub_boxes, obj_boxes, sub_center_points, obj_center_points = [], [], [], [], [], []
    if dataset_type.value==1:
        hoi_labels = []
    sub_obj_pairs = []
    for hoi in annotation['hoi_annotation']:
        sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
        if sub_obj_pair in sub_obj_pairs:
        # 已经存在的pair就直接在类别+1即可，说明已经写入了
            verb_labels[sub_obj_pairs.index(sub_obj_pair)][hoi['category_id']] = 1
            if dataset_type.value==1:
                hoi_labels[sub_obj_pairs.index(sub_obj_pair)][hoi['hoi_category_id']] = 1
        else:
            sub_obj_pairs.append(sub_obj_pair)
            if hoi['object_id']==-1:
                obj_labels.append(torch.as_tensor(1, dtype=torch.int64))
            else:
                obj_labels.append(target['labels'][hoi['object_id']])
            # 当为HICO-DET数据集时，id从1开始，则列表首元素下表为0不会被涉及，需要额外+1
            if dataset_type.value==1:
                hoi_label = [0 for _ in range(HICO_DET_HOI_NUM+1)] 
                hoi_label[hoi['hoi_category_id']] = 1
                verb_label = [0 for _ in range(HICO_DET_VERB_NUM+1)]
            # 当为V-COCO数据集时，id从0开始，无需额外操作
            else:
                verb_label = [0 for _ in range(V_COCO_VERB_NUM)]
            verb_label[hoi['category_id']] = 1
            # verb是哪个行为就在对应下标的数据打1,同理hoi
            sub_box = target['boxes'][hoi['subject_id']]
            sub_center_point = target['center_points'][hoi['subject_id']]
            if hoi['object_id']==-1:
                obj_box = torch.as_tensor([0, 0, 0, 0], dtype=torch.float32)
                obj_center_point = torch.as_tensor([0, 0], dtype=torch.float32)
            else:
                obj_box = target['boxes'][hoi['object_id']]
                obj_center_point = target['center_points'][hoi['object_id']] 
            verb_labels.append(verb_label)
            if dataset_type.value==1:
                hoi_labels.append(hoi_label)
            sub_boxes.append(sub_box) 
            obj_boxes.append(obj_box)
            sub_center_points.append(sub_center_point) 
            obj_center_points.append(obj_center_point)
            
    # 遍历所有HOI后对target进行重新堆叠
    if len(sub_obj_pairs) == 0:
        target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
        if dataset_type.value==1:
            target['verb_labels'] = torch.zeros((0, HICO_DET_VERB_NUM+1), dtype=torch.float32)
            target['hoi_labels'] = torch.zeros((0, HICO_DET_HOI_NUM+1), dtype=torch.float32)
        else:
            target['verb_labels'] = torch.zeros((0, V_COCO_VERB_NUM), dtype=torch.float32)
        target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32) 
        target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        target['sub_center_points'] = torch.zeros((0, 2), dtype=torch.float32) 
        target['obj_center_points'] = torch.zeros((0, 2), dtype=torch.float32)
    else:
        target['obj_labels'] = torch.stack(obj_labels)
        target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
        if dataset_type.value==1:
            target['hoi_labels'] = torch.as_tensor(hoi_labels, dtype=torch.float32)
        target['sub_boxes'] = torch.stack(sub_boxes)
        target['obj_boxes'] = torch.stack(obj_boxes)
        target['sub_center_points'] = torch.stack(sub_center_points)
        target['obj_center_points'] = torch.stack(obj_center_points)
    
    return target


# 输入图像（image）、解析后的标注信息列表（target）、数据集类型标注（dataset_type）和简单数据增强对应的TransformType（trans_type）
# 输出一张增强后的图片（image_trans）及其信息列表（target_trans）
def transImageAndAnno(image, target, dataset_type, trans_type):
    height = target['height']
    width = target['width']
    if trans_type.value<4:
        if trans_type.value==1:
            transform = T.HFlipTransform(width)
        elif trans_type.value==2:
            transform = T.VFlipTransform(height)
        # 仅剩value为3的情况
        else:
            transform = T.NoOpTransform()
        image_trans = transform.apply_image(image)  # new image
        boxes_trans = torch.as_tensor(transform.apply_box(target['boxes']), dtype=torch.float32).reshape(-1, 4)
        sub_boxes_trans = torch.as_tensor(transform.apply_box(target['sub_boxes']), dtype=torch.float32).reshape(-1, 4)
        obj_boxes_trans = torch.as_tensor(transform.apply_box(target['obj_boxes']), dtype=torch.float32).reshape(-1, 4)
    else:
        if trans_type.value==4:
            augs = T.AugmentationList([T.RandomCrop("relative_range", (0.8,0.8))])
        elif trans_type.value==5:
            augs = T.AugmentationList([T.RandomExtent((1.2, 1.2), (0.2,0.2))])
        elif trans_type.value==6:
            augs = T.AugmentationList([T.RandomRotation(0.8, 1.2)])
        elif trans_type.value==7:
            augs = T.AugmentationList([T.RandomBrightness(0.8, 1.2)])
        # 仅剩value为8的情况
        else:
            augs = T.AugmentationList([T.RandomContrast(0.8, 1.2)])
        integrated_boxes = np.concatenate((target['boxes'].numpy(), target['sub_boxes'].numpy(), target['obj_boxes'].numpy()), axis=0)
        input = T.AugInput(image, boxes=integrated_boxes)
        transform = augs(input)  
        image_trans = input.image
        integrated_boxes_trans = input.boxes
        
        boxes_trans = torch.as_tensor(integrated_boxes_trans[:target['boxes'].shape[0],:], dtype=torch.float32).reshape(-1, 4)
        sub_boxes_trans = torch.as_tensor(integrated_boxes_trans[target['boxes'].shape[0]:target['boxes'].shape[0]+target['sub_boxes'].shape[0],:], dtype=torch.float32).reshape(-1, 4)
        obj_boxes_trans = torch.as_tensor(integrated_boxes_trans[target['boxes'].shape[0]+target['sub_boxes'].shape[0]:, :], dtype=torch.float32).reshape(-1, 4)
    
    # 图像（image）已经增强完成，但需要对一些数据进行重新处理，重新生成target
    target_trans = target.copy()
    height_trans = image_trans.shape[0]
    width_trans = image_trans.shape[1]
    target_trans['height'] = torch.as_tensor(int(height_trans))
    target_trans['width'] = torch.as_tensor(int(width_trans))

    # 防止新产生的boxes范围超出图像范围
    boxes_trans[:, 0::2].clamp_(min=0, max=width_trans)
    boxes_trans[:, 1::2].clamp_(min=0, max=height_trans)
    sub_boxes_trans[:, 0::2].clamp_(min=0, max=width_trans)
    sub_boxes_trans[:, 1::2].clamp_(min=0, max=height_trans)
    obj_boxes_trans[:, 0::2].clamp_(min=0, max=width_trans)
    obj_boxes_trans[:, 1::2].clamp_(min=0, max=height_trans)

    # 根据数据坐标比较，保留合法的box，由于可能会有操作导致一些bbox被挤出图像之外，因此不仅需要删除bbox，还要删除对应的label、verb与hoi列表
    keep_trans = (boxes_trans[:, 3] > boxes_trans[:, 1]) & (boxes_trans[:, 2] > boxes_trans[:, 0]) 
    target_trans['boxes'] = boxes_trans[keep_trans]
    target_trans['labels'] = target_trans['labels'][keep_trans]
    sub_keep_trans = (sub_boxes_trans[:, 3] > sub_boxes_trans[:, 1]) & (sub_boxes_trans[:, 2] > sub_boxes_trans[:, 0])
    zeros = torch.zeros(obj_boxes_trans.shape[0], dtype=torch.int64)
    obj_keep_trans = (((obj_boxes_trans[:, 3] > obj_boxes_trans[:, 1]) & (obj_boxes_trans[:, 2] > obj_boxes_trans[:, 0])) | ((obj_boxes_trans[:, 0] == zeros) & (obj_boxes_trans[:, 1] == zeros) & (obj_boxes_trans[:, 2] == zeros) & (obj_boxes_trans[:, 3] == zeros)))
    integrated_keep_trans = sub_keep_trans & obj_keep_trans
    target_trans['sub_boxes'] = sub_boxes_trans[integrated_keep_trans]
    target_trans['obj_boxes'] = obj_boxes_trans[integrated_keep_trans]
    target_trans['obj_labels'] = target_trans['obj_labels'][integrated_keep_trans]
    target_trans['verb_labels'] = target_trans['verb_labels'][integrated_keep_trans]
    if(dataset_type.value==1):
        target_trans['hoi_labels'] = target_trans['hoi_labels'][integrated_keep_trans]

    # center_points部分需要完全重新计算
    target_trans['center_points'] = torch.mean(target_trans['boxes'].reshape(-1, 2, 2),1).reshape(-1, 2)
    target_trans['sub_center_points'] = torch.mean(target_trans['sub_boxes'].reshape(-1, 2, 2),1).reshape(-1, 2)
    target_trans['obj_center_points'] = torch.mean(target_trans['obj_boxes'].reshape(-1, 2, 2),1).reshape(-1, 2)

    return image_trans, target_trans


# 输入解析后的标注信息列表（target）、指定的文件名称、数据集类型标注（dataset_type）和若为HICO-DET应当再次指定annotation中的img_id（它从1开始，V_COCO可指定为0）
# 输出原json文件形式的标注（annotation）
def targetToAnno(target, file_name, dataset_type, img_id):
    annotation = {}
    annotation['file_name'] = file_name
    if dataset_type.value==1:
        annotation['img_id'] = img_id
        bounding_box_list,hoi_detail_list = [], []
        for box, category_id in zip(target['boxes'].type(torch.int64).tolist(), target['labels'].tolist()):
            bounding_box_item = {}
            bounding_box_item['bbox'] = box
            bounding_box_item['category_id'] = category_id
            bounding_box_list.append(bounding_box_item)
        annotation['annotation'] = bounding_box_list
        for sub_box, obj_box, verb_list, hoi_category_list in zip(target['sub_boxes'].type(torch.int64).tolist(), target['obj_boxes'].type(torch.int64).tolist(), target['verb_labels'].tolist(), target['hoi_labels'].tolist()):
            verb_ids = [index for index, value in enumerate(verb_list) if value==1]
            hoi_categories = [index for index, value in enumerate(hoi_category_list) if value==1]
            subject_id = target['boxes'].type(torch.int64).tolist().index(sub_box)
            object_id = target['boxes'].type(torch.int64).tolist().index(obj_box)
            for verb_id, hoi_category in zip(verb_ids, hoi_categories):
                hoi_detail_item = {}
                hoi_detail_item['subject_id'] = subject_id
                hoi_detail_item['object_id'] = object_id
                hoi_detail_item['category_id'] = verb_id
                hoi_detail_item['hoi_category_id'] = hoi_category
                hoi_detail_list.append(hoi_detail_item)
        annotation['hoi_annotation'] = hoi_detail_list
    else:
        bounding_box_list,hoi_detail_list = [], []
        for sub_box, obj_box, verb_list in zip(target['sub_boxes'].tolist(), target['obj_boxes'].tolist(), target['verb_labels'].tolist()):
            verb_ids = [index for index, value in enumerate(verb_list) if value==1]
            subject_id = target['boxes'].tolist().index(sub_box)
            if(obj_box == [0, 0, 0, 0]):
                object_id = -1
            else:
                object_id = target['boxes'].tolist().index(obj_box)
            for verb_id in verb_ids:
                hoi_detail_item = {}
                hoi_detail_item['subject_id'] = subject_id
                hoi_detail_item['category_id'] = verb_id
                hoi_detail_item['object_id'] = object_id
                hoi_detail_list.append(hoi_detail_item)
        annotation['hoi_annotation'] = hoi_detail_list
        for box, category_id in zip(target['boxes'].tolist(), target['labels'].tolist()):
            bounding_box_item = {}
            bounding_box_item['category_id'] = category_id
            bounding_box_item['bbox'] = box
            bounding_box_list.append(bounding_box_item)
        annotation['annotation'] = bounding_box_list
        
    return annotation


# 输入标注的json文件路径（json_path）、所需要的标注（annotation）在数组中的下标（index）、存放图片的文件夹的路径（image_folder_path）、一个数据集类型的标识（dataset_type）和简单数据增强对应的TransformType（trans_type）
# 输出一张增强后的图片（image_trans）及其解析后的标注信息列表（target_trans）
# 可以通过上述targetToAnno函数将其转化为对应的注释，但要提供新的file_name和dataset_type
def simpleAug(json_path, index, image_folder_path, dataset_type, trans_type):
    annotation = getAnno(json_path, index)
    if annotation== False:
        error_message = "no such index in annotation file"
        return False, error_message
    image = getImage(image_folder_path, annotation)
    target = analyseAnno(image, annotation, dataset_type)
    image_trans, target_trans = transImageAndAnno(image, target, dataset_type, trans_type)
    return image_trans, target_trans