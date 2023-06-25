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
from math import pi

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.data import transforms as T


TransformDictionary = {
    0 : T.RandomFlip(1, horizontal=True, vertical=False),
    1 : T.RandomFlip(1, horizontal=False, vertical=True),
    2 : T.RandomCrop("relative_range", (0.8,0.8)),
    3 : T.RandomExtent((1.2, 1.2), (0.2,0.2)),
    4 : T.RandomRotation([1*pi, 2*pi], center=[[0.4, 0.4], [0.6, 0.6]], sample_style='range'),
    5 : T.RandomBrightness(0.8, 1.2),
    6 : T.RandomContrast(0.8, 1.2)
}


'''对于最简单的数据增强，一共有7种增强变换HFlipTransform、VFlipTransform、RandomCrop、RandomExtent、RandomRotation、RandomBrightness、RandomContrast，其中HFlipTransform、VFlipTransform可以全部使用randomFlip完成，这样所有的操作可以使用augmentationList集中存放，内部数据固定不调整，变化只取决于哪些会被采用'''
def simpleAug(image, width, height, boxes):
    # 有1/10的概率进行数据增强，此时产生的随机数为0，对应的isAug为true
    isAug = (torch.randint(0,10,(1,)).item() == 0)
    if isAug:
        # 对于每一个增强选项均有概率参与数据增强，为了保证至少有一项参与数据增强，因此，在所有选项均为false时，平等地随机选择一项进行数据增强
        # 我们希望有进行多项增强的可能性，但是进行单项数据增强的概率最高，而单项概率越低，这个值越高，我们将单项概率设为0.1，这样约有85%概率进行单项增强
        augChoice = torch.randint(0,10,(7,)) == torch.zeros([7])
        if(not augChoice.sum()):
            augChoice[torch.randint(0,7,(1,)).item()] = True
        AugList = []
        for index in range(7):
            if augChoice[index].item():
                AugList.append(TransformDictionary.get(index))
        augs = T.AugmentationList(AugList)
        image = np.array(image)
        input = T.AugInput(image, boxes=boxes.numpy()) 
        image_trans = Image.fromarray(input.image.astype('uint8'))
        boxes_trans = torch.as_tensor(input.boxes, dtype=torch.float32).reshape(-1, 4)
        width_trans, height_trans = image_trans.size
    else:
        return image, width, height, boxes
    return image_trans, width_trans, height_trans, boxes_trans