# Character-object interaction detection task based on environmental perception
This paper is developed based on QPIC: Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information by [Masato Tamura](https://scholar.google.co.jp/citations?user=IbPzCocAAAAJ), [Hiroki Ohashi](https://scholar.google.com/citations?user=GKC6bbYAAAAJ), and Tomoaki Yoshinaga which implemented by expanding [DETR](https://github.com/hitachi-rd-cv/qpic). You can check the original QPIC code from [this respository](https://github.com/hitachi-rd-cv/qpic).

Based on QPIC, This paper develops 3 methods to improve the model performance: simple data augmentation, middle data augmentation and environment sensor module.
by [Zengyu Ye](spoilvoid.github.io).

## metrics explanation

![Data Augmentation Procedure ](figure/data_augmentation_procedure.png)
### Simple Augmentation
Simple augmentation uses [Detectron2](https://github.com/facebookresearch/detectron2) to conduct 7 basic figure transforms. There names and operation functions are as follows:
1)	Horizontal Flip: `RandomFlip(1,horizontal=True,vertical=False)`
2)	Vertical Flip: `RandomFlip(1,horizontal=False,vertical=True)`
3)	Random Vrop: `RandomCrop("relative_range",(0.8,0.8))`
4) Random Extent: `RandomExtent((1.2,1.2),(0.2,0.2))`
5)	Random Rotation: `RandomRotation([π,2pi], center=[[0.4,0.4],[0.6,0.6]],sample_style=’range’)`
6)	Random Brightness: `RandomBrightness(0.8,1.2)`
7)	Random Contrast: `RandomContrast(0.8,1.2)`

### Middle Augmentation
Middle augmentation move person or object in low distances like shift with standard Gassuain Distribution and then uses FMM(Fast Marching Method) algorithm in *opencv-python* to fix missing pixels like `cv2.inpaint(source,inpaintMask, inpaintRadius=5, ﬂags=cv2.INPAINT_TELEA)`

### Environment Sensor
Original QPIC only judges HOI behaviors based on 2 bounding boxs. However, background will influence behaviors types, like we intend to sleep in the bedroom. Thus we add a *SENet* like architecture to aid Transformer Encoder to judge HOI behaviors like below.
![Data Augmentation Procedure ](figure/qpic_modify.png)

Detailed *Environment Sensor* Architectures are as follows.
![Data Augmentation Procedure ](figure/environment_sensor.png)

## Preparation

### Dependencies
Our implementation uses external libraries such as NumPy and PyTorch. You can resolve the dependencies with the following command.
```
pip install numpy
pip install -r requirements.txt
```
Note that this command may dump errors during installing pycocotools, but the errors can be ignored.

To run code for 3 extra method. You can resolve the dependencies with the following command.
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
chmod +x Anaconda3-2023.03-Linux-x86_64.sh
./Anaconda3-2023.03-Linux-x86_64.sh
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install matplotlib --force-reinstall
pip install opencv-python
pip install fvcore
pip install ninja
pip install cython
pip install pycocotools
pip install submitit
pip install git+https://github.com/cocodataset/panopticapi.git
pip install scipy
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Dataset

#### HICO-DET
HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.
```
hoi_env_aug
 |─ data
 │   └─ hico_20160224_det
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |   └─ corre_hico.npy
 :       :
```

#### V-COCO
First clone the repository of V-COCO from [here](https://github.com/s-gupta/v-coco), and then follow the instruction to generate the file `instances_vcoco_all_2014.json`. Next, download the prior file `prior.pickle` from [here](https://drive.google.com/drive/s/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4). Place the files and make directories as follows.
```
hoi_env_aug
 |─ data
 │   └─ v-coco
 |       |─ data
 |       |   |─ instances_vcoco_all_2014.json
 |       |   :
 |       |─ prior.pickle
 |       |─ images
 |       |   |─ train2014
 |       |   |   |─ COCO_train2014_000000000009.jpg
 |       |   |   :
 |       |   └─ val2014
 |       |       |─ COCO_val2014_000000000042.jpg
 |       |       :
 |       |─ annotations
 :       :
```
For our implementation, the annotation file have to be converted to the HOIA format. The conversion can be conducted as follows.
```
PYTHONPATH=data/v-coco \
        python convert_vcoco_annotations.py \
        --load_path data/v-coco/data \
        --prior_path data/v-coco/prior.pickle \
        --save_path data/v-coco/annotations
```
Note that only Python2 can be used for this conversion because `vsrl_utils.py` in the v-coco repository shows a error with Python3.

V-COCO annotations with the HOIA format, `corre_vcoco.npy`, `test_vcoco.json`, and `trainval_vcoco.json` will be generated to `annotations` directory.

### Pre-trained parameters
QPIC has to be pre-trained with the COCO object detection dataset. For the HICO-DET training, this pre-training can be omitted by using the parameters of DETR. The parameters can be downloaded from [here](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth) for the ResNet50 backbone, and [here](https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth) for the ResNet101 backbone. For the V-COCO training, this pre-training has to be carried out because some images of the V-COCO evaluation set are contained in the training set of DETR. You have to pre-train QPIC without those overlapping images by yourself for the V-COCO evaluation. We offer you additional V-COCO pretrained model weight from [here](https://drive.google.com/file/d/12RnfCpfAmAN089StR29h2UJ7qS96bIsq/view?usp=drive_link) for the ResNet50 backbone. You can also download HICO-DET trained model wight from the same  for the ResNet50 backbone.

For HICO-DET, move the downloaded parameters to the `params` directory and convert the parameters with the following command.
```
python convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-hico.pth
```

For V-COCO, convert the pre-trained parameters with the following command.
```
python convert_parameters.py \
        --load_path logs/checkpoint.pth \
        --save_path params/detr-r50-pre-vcoco.pth \
        --dataset vcoco
```

## Training
After the preparation, you can start the training with the following command.

For the basic QPIC HICO-DET training.
```
python main.py \
        --pretrained params/detr-r50-pre-hico.pth \
        --output_dir logs\hico_det \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1
```
If you want to train the model with *simple augmentation*, set `--data_augmentation simple`. If you want to train the model with *middle augmentation*, set `--data_augmentation middle`. If you want to train the model with *environment sensor*, set `--environment yes`.
For the simple augmentation QPIC HICO-DET training.

For the basic QPIC V-COCO training.
```
python main.py \
        --pretrained params/detr-r50-pre-vcoco.pth \
        --output_dir logs\vcoco \
        --hoi \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1
```
The other settings are the same as HICO-DET training.

Note that the number of object classes is 81 because one class is added for missing object.

If you have multiple GPUs on your machine, you can utilize them to speed up the training. The number of GPUs is specified with the `--nproc_per_node` option. The following command starts the training with 8 GPUs for the HICO-DET training.
```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --pretrained params/detr-r50-pre-hico.pth \
        --output_dir logs \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1
```

### Trained parameters
The below trained model parameters are available [here](https://drive.google.com/drive/s/1C499l5S7UIip2VJhw7h48NYjI7Rruo07?usp=drive_link). The *logs* folder contains train parameters and corresponding evaluation results and *models* folder contains each trained model with the last epoch.

## Evaluation
The evaluation is conducted at the end of each epoch during the training. The results are written in `logs/log.txt` like below:
```
"test_mAP": 0.29061250833779456, "test_mAP rare": 0.21910348492395765, "test_mAP non-rare": 0.31197234650036926
```
`test_mAP`, `test_mAP rare`, and `test_mAP non-rare` are the results of the default full, rare, and non-rare setting, respectively.

You can also conduct the evaluation with trained parameters as follows.
```
python main.py \
        --pretrained detr_resnet50_hico.pth \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --eval
```

For the official evaluation of V-COCO, a pickle file of detection results have to be generated. You can generate the file as follows.
```
python generate_vcoco_official.py \
        --param_path logs/checkpoint.pth \
        --save_path vcoco.pickle \
        --hoi_path data/v-coco
```

## Results
***Our result is only trained in the ResNet50 backbone, thus we'll only compare the ResNet50 backbone below.***

HICO-DET.
| models | mAP(full) | mAP(none-rare) | mAP(rare) | max recall |
| :--- | :---: | :---: | :---: | :---: |
| self-trained QPIC | 25.89 | 28.03 | 18.74 | 57.62 |
| QPIC + simple augmentation | 27.02 | 29.15 | 20.23 | 59.51 |
| QPIC + middle augmentation | 26.85 | 29.11 | 20.44 | 59.26 |
| QPIC + environment sensor | 26.77 | 28.76 | 20.43 | 59.58 |
| original QPIC(ResNet101) | 29.07 | 27.42 | 31.69 | / |

V-COCO.
|models | mAP |
| :--- | :---: |
| self-trained QPIC | 54.79 |
| QPIC + simple augmentation | 56.84 |
| QPIC + middle augmentation | 55.33 |
| QPIC + environment sensor | 57.04 |
| original QPIC(ResNet101) | 58.8 |
```
