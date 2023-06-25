# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

# argparse是一个Python模块：命令行选项、参数和子命令解析器
# 程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数
import argparse
import random
from pathlib import Path
from PIL import Image
import itertools
import copy
import json
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as standard_transforms

import util.misc as utils
from datasets import build_dataset
from engine import HICOEvaluator, VCOCOEvaluator
from models import build_model

class imageset(Dataset):
    def __init__(self, img_folder, anno_file):
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self.ids = list(range(len(self.annotations)))
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        img = np.array(img)

        return img


def get_args_parser():
    # 创建一个 ArgumentParser 对象
    # ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
    # description - Text to display before the argument help (by default, no text)
    # add_help - 为解析器添加一个 -h/--help 选项（默认值：True）
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # 给一个 ArgumentParser 添加程序参数信息（这里是模型超参数）
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters（预训练权重参数）设置该参数只有mask头会改变
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone（DETR的backbone选择r50还是r101）
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--branch_CNN', default='resnet50', type=str,
                        help="Name of the branch CNN to use")
    # 最后一个卷积块中使用膨胀卷积（dilated convolution）还是步幅卷积（stride convolution）
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # 在图像特征之上使用的位置嵌入（positional embedding）类型
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # 在图像特征之上使用的位置嵌入（positional embedding）类型
    parser.add_argument('--data_augmentation', default='none', type=str, choices=('none', 'simple', 'middle'),
                        help="Type of data augmentation to use on every image")
    parser.add_argument('--environment', default='no', type=str, choices=('no', 'yes'),
                        help="whether to use environment block in the model")
    return parser

def main(args):
    # 根据参数判断使用gpu还是cpu，产生设备（device）对象
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # utils.get_rank()当分布式训练时，需要多个seed，用它获取当前的排序加上种子以此使得所有分布式设备的seed都不同
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, branch_criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    print("model created")
    # 测算参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico.json')
    }
    img_folder, anno_file = PATHS['train']
    
    '''使用datasets文件夹中的__init__.py中的总build_dataset函数'''
    '''可以修改此处进行完整的数据增强后产生的数据集的训练'''
    dataset_val = build_dataset(image_set='val', args=args)
    imageset_val = build_imageset(image_set='val', args=args)

    # 训练样本使用随机采样，评估样本顺序采样
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # collate_fn定义了如何取样本的，我们可以定义自己的函数来准确地实现想要的功能。
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    image_loader_val = DataLoader(imageset_val, drop_last=False)

    print("dataloader created")

    # 对整个网络进行训练时，从resume或pretrained载入权重
    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    print("trained weight loaded")
    images, true_preds, false_preds, gts = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, image_loader_val, args.subject_category_id, device)

    if (args.dataset_file == 'hico'):
        visualize_preds(images, true_preds, false_preds, args.num_verb_classes + 1)
        visualize_gts(images, gts, args.num_verb_classes + 1)
    elif (args.dataset_file == 'vcoco'):
        visualize_preds(images, true_preds, false_preds, args.num_verb_classes)
        visualize_gts(images, gts, args.num_verb_classes)

def build_imageset(image_set, args):
    if args.dataset_file == 'hico':
        root = Path(args.hoi_path)
        assert root.exists(), f'provided HOI path {root} does not exist'
        PATHS = {
            'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico.json'),
            'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico.json')
        }
        img_folder, anno_file = PATHS[image_set]
        dataset = imageset(img_folder, anno_file)
        return dataset
    if args.dataset_file == 'vcoco':
        root = Path(args.hoi_path)
        assert root.exists(), f'provided HOI path {root} does not exist'
        PATHS = {
            'train': (root / 'images' / 'train2014', root / 'annotations' / 'trainval_vcoco.json'),
            'val': (root / 'images' / 'val2014', root / 'annotations' / 'test_vcoco.json')
        }
        img_folder, anno_file = PATHS[image_set]
        dataset = imageset(img_folder, anno_file)
        return dataset
    raise ValueError(f'dataset {args.dataset_file} not supported')

@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, image_loader, subject_category_id, device):
    model.eval()

    preds = []
    gts = []
    indices = []

    # 这里将从dataloader中一个batch地取image与target，完整地处理整个数据集，产生preds和gts
    for index, (samples, targets) in enumerate(data_loader):
        if index >= 3:
            break
        samples = samples.to(device)

        outputs, _ = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))
        images, masks = samples.decompose()
        image_list = [np.array(standard_transforms.ToPILImage()(image)) for image in images]
        image_list = np.array(image_list)
    images = []
    for index, image in enumerate(image_loader):
        if index >= 6:
            break
        images.append(image)


    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat)

    evaluator.judgePreds()
    true_preds, false_preds, gts = evaluator.getPredsAndGts()

    return images, true_preds, false_preds, gts

def visualize_preds(images, true_annotations, false_annotations, VERB_NUM):
    count = 1
    for image, true_annotation, false_annotation in zip(images, true_annotations, false_annotations):
        fig, ax = plt.subplots(1, 1) 
        axis = fig.gca()
        plt.imshow(image[0])
        # 正确的hoi记录
        cnt = 0
        '''for hoi in true_annotation:
            cnt = cnt + 1
            if cnt >= 5:
                break
            # 从一条hoi记录中获取所需信息
            sub_box = hoi['sub_box']['bbox']
            sub_category_id = hoi['sub_box']['category_id']
            obj_box = hoi['obj_box']['bbox']
            obj_category_id = hoi['obj_box']['category_id']
            hoi_category_id = hoi['category_id']
            score = hoi['score']
            sub_center_point = (np.mean(sub_box[0::2]),np.mean(sub_box[1::2]))
            obj_center_point = (np.mean(obj_box[0::2]),np.mean(obj_box[1::2]))
            # 画subject与object界限框
            sub_rec = plt.Rectangle((sub_box[0], sub_box[1]), width=sub_box[2]-sub_box[0], height=sub_box[3]-sub_box[1],fill=False, edgecolor = 'blue',linewidth=3)
            axis.add_patch(sub_rec)
            if(obj_category_id != -1):
                obj_rec = plt.Rectangle((obj_box[0], obj_box[1]), width=obj_box[2]-obj_box[0], height=obj_box[3]-obj_box[1],fill=False, edgecolor = 'red',linewidth=3)
                axis.add_patch(obj_rec)
            # 画表示hoi的线段与文本
            if(obj_category_id != -1):
                plt.plot([sub_center_point[0], obj_center_point[0]], [sub_center_point[1], obj_center_point[1]], linewidth=4, color="green")
                line_center_points = ((sub_center_point[0]+obj_center_point[0])/2, (sub_center_point[1]+obj_center_point[1])/2)
            else:
                line_center_points = sub_center_point'''
        # 错误的hoi记录
        for hoi in false_annotation:
            cnt = cnt + 1
            if cnt >= 5:
                break
            # 从一条hoi记录中获取所需信息
            sub_box = hoi['sub_box']['bbox']
            sub_category_id = hoi['sub_box']['category_id']
            obj_box = hoi['obj_box']['bbox']
            obj_category_id = hoi['obj_box']['category_id']
            hoi_category_id = hoi['category_id']
            score = hoi['score']
            sub_center_point = (np.mean(sub_box[0::2]),np.mean(sub_box[1::2]))
            obj_center_point = (np.mean(obj_box[0::2]),np.mean(obj_box[1::2]))
            # 画subject与object界限框
            sub_rec = plt.Rectangle((sub_box[0], sub_box[1]), width=sub_box[2]-sub_box[0], height=sub_box[3]-sub_box[1],fill=False, edgecolor = 'gold',linewidth=3)
            axis.add_patch(sub_rec)
            if(obj_category_id != -1):
                obj_rec = plt.Rectangle((obj_box[0], obj_box[1]), width=obj_box[2]-obj_box[0], height=obj_box[3]-obj_box[1],fill=False, edgecolor = 'purple',linewidth=3)
                axis.add_patch(obj_rec)
            # 画表示hoi的线段与文本
            if(obj_category_id != -1):
                plt.plot([sub_center_point[0], obj_center_point[0]], [sub_center_point[1], obj_center_point[1]], linewidth=4, color="black")
                line_center_points = ((sub_center_point[0]+obj_center_point[0])/2, (sub_center_point[1]+obj_center_point[1])/2)
            else:
                line_center_points = sub_center_point
            
        # 图像保存
        image_name = './figure/preds_' + str(count).zfill(8) +'.jpg'
        plt.savefig(image_name)
        plt.cla()
        plt.clf()
        count = count + 1

def visualize_gts(images, annotations, VERB_NUM):
    count = 1
    for image, annotation in zip(images, annotations):
        image = np.array(image[0])
        boxes = torch.as_tensor(np.array([obj['bbox'] for obj in annotation['annotations']]), dtype=torch.float32).reshape(-1, 4)
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
        sub_obj_pairs = []
        for hoi in annotation['hoi_annotation']:
            sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
            if sub_obj_pair in sub_obj_pairs:
            # 已经存在的pair就直接在类别+1即可，说明已经写入了
                verb_labels[sub_obj_pairs.index(sub_obj_pair)][hoi['category_id']] = 1
            else:
                sub_obj_pairs.append(sub_obj_pair)
                if hoi['object_id']==-1:
                    obj_labels.append(torch.as_tensor(-1, dtype=torch.int64))
                else:
                    obj_labels.append(target['labels'][hoi['object_id']])
                # verb是哪个行为就在对应下标的数据打1
                verb_label = [0 for _ in range(VERB_NUM)]
                verb_label[hoi['category_id']] = 1
                
                sub_box = target['boxes'][hoi['subject_id']]
                sub_center_point = target['center_points'][hoi['subject_id']]
                if hoi['object_id']==-1:
                    obj_box = torch.as_tensor([0, 0, 0, 0], dtype=torch.float32)
                    obj_center_point = torch.as_tensor([0, 0], dtype=torch.float32)
                else:
                    obj_box = target['boxes'][hoi['object_id']]
                    obj_center_point = target['center_points'][hoi['object_id']] 
                verb_labels.append(verb_label)
                sub_boxes.append(sub_box) 
                obj_boxes.append(obj_box)
                sub_center_points.append(sub_center_point) 
                obj_center_points.append(obj_center_point)
                
        # 遍历所有HOI后对target进行重新堆叠
        if len(sub_obj_pairs) == 0:
            target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
            target['verb_labels'] = torch.zeros((0, VERB_NUM), dtype=torch.float32)
            target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32) 
            target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['sub_center_points'] = torch.zeros((0, 2), dtype=torch.float32) 
            target['obj_center_points'] = torch.zeros((0, 2), dtype=torch.float32)
        else:
            target['obj_labels'] = torch.stack(obj_labels)
            target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
            target['sub_boxes'] = torch.stack(sub_boxes)
            target['obj_boxes'] = torch.stack(obj_boxes)
            target['sub_center_points'] = torch.stack(sub_center_points)
            target['obj_center_points'] = torch.stack(obj_center_points)

        # 图像生成  
        fig, ax = plt.subplots(1, 1) 
        axis = fig.gca()
        plt.imshow(image)
        for sub_box, obj_box in zip(target['sub_boxes'], target['obj_boxes']):
            sub_rec = plt.Rectangle((sub_box[0].item(), sub_box[1].item()), width=sub_box[2].item()-sub_box[0].item(), height=sub_box[3].item()-sub_box[1].item(),fill=False, edgecolor = 'blue',linewidth=3)
            obj_rec = plt.Rectangle((obj_box[0].item(), obj_box[1].item()), width=obj_box[2].item()-obj_box[0].item(), height=obj_box[3].item()-obj_box[1].item(),fill=False, edgecolor = 'red',linewidth=3)
            axis.add_patch(sub_rec)
            axis.add_patch(obj_rec)  
        for sub_point, obj_point, obj_label, verb_label in zip(target['sub_center_points'],  target['obj_center_points'], target['obj_labels'], target['verb_labels']):
            if(obj_label != -1):
                plt.plot([sub_point[0].item(), obj_point[0].item()], [sub_point[1].item(), obj_point[1].item()], linewidth=4, color="green")
                center_points = ((sub_point[0].item()+obj_point[0].item())/2, (sub_point[1].item()+obj_point[1].item())/2)
            else:
                center_points = (sub_point[0].item(), sub_point[1].item())
            indices = [index for index, value in enumerate(verb_labels) if value == 1]
        image_name = './figure/gts_' + str(count).zfill(8) +'.jpg'
        plt.savefig(image_name)
        plt.cla()
        plt.clf()
        count = count + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
