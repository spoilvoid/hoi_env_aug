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
import datetime
import json
import random
import time
import os
import sys
import multiprocessing
from pathlib import Path

import numpy as np
import torch, gc
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, evaluate_hoi
from models import build_model


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
    # * Branch_CNN（DETR的branch CNN选择r50还是r101）
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

    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    multiprocessing.set_start_method('spawn')
    # 判断是否进行分布式训练，根据你的电脑的环境配置中是否有相关配置来判断或设置
    # 将当前默认的错误输出结果保存为__stderr__
    #__stderr__ = sys.stderr 
    #sys.stderr = open('errorlog.txt', 'a') 
    utils.init_distributed_mode(args)
    # utils.get_sha()通过命令行获得git的commit ID和git status以及所在的branch。
    print("git:\n  {}\n".format(utils.get_sha()))

    # 如果已经载入了预训练参数，冻结训练只使用于分割
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # 根据参数判断使用gpu还是cpu，产生设备（device）对象
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # utils.get_rank()当分布式训练时，需要多个seed，用它获取当前的排序加上种子以此使得所有分布式设备的seed都不同
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, branch_criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        # 在分布式模型中并行地运行模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # 测算参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # 所有参数组成的字典
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # 优化器选择朴素的AdamW
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # 使用StepLR每lr_drop个epoch后下降，下降因子为默认的0.1
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    '''使用datasets文件夹中的__init__.py中的总build_dataset函数'''
    '''可以修改此处进行完整的数据增强后产生的数据集的训练'''
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        # 训练样本使用随机采样，评估样本顺序采样
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # drop_last为true时，如果最后一批的数量少于batch_size，则抛弃
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # collate_fn定义了如何取样本的，我们可以定义自己的函数来准确地实现想要的功能。
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if not args.hoi:
        if args.dataset_file == "coco_panoptic":
            # We also evaluate AP during panoptic training, on original coco DS
            coco_val = datasets.coco.build("val", args)
            base_ds = get_coco_api_from_dataset(coco_val)
        else:
            base_ds = get_coco_api_from_dataset(dataset_val)

    # 如果只训练mask头，则从frozen_weights地址向模型载入预训练权重
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # 模型输出设置
    output_dir = Path(args.output_dir)
    # 对整个网络进行训练时，从resume或pretrained载入权重
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    # 是否进行模型评估
    if args.eval:
        if args.hoi:
            test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device)
            return
        else:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 清除无用缓存
        #gc.collect()
        #torch.cuda.empty_cache()
        if args.distributed:
            # 在分布式模式下，需要在每个 epoch 开始时调用 set_epoch() 方法，然后再创建 DataLoader 迭代器，以使 shuffle 操作能够在多个 epoch 中正常工作。 否则，dataloader迭代器产生的数据将始终使用相同的顺序。
            sampler_train.set_epoch(epoch)
        # 调用engine中的训练函数，来执行具体的损失函数计算，梯度计算等工作
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, branch_criterion)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if args.hoi:
            test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device)
            coco_evaluator = None
        else:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch, 
                    'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
