# TensorFlow implementation of DARK Pose
This is an unofficial TensorFlow implementation of DARK Pose ([Distribution Aware Coordinate Representation for Human Pose Estimation](https://arxiv.org/abs/1910.06278)).

It is based on the official PyTorch implementation [ilovepose/DarkPose](https://github.com/ilovepose/DarkPose).

## Requirements
- Python 3.7
- TensorFlow 2.1

## Installation
1. Create an anaconda environment.
```sh
conda create -n tf-dark-pose python=3.7 anaconda
```

2. Activate the environment.
```sh
conda activate tf-dark-pose
```

3. Install pip packages.
```sh
pip install -r requirements.txt
```

## Training on COCO
1. Download the images and annotation files (2017 train/val) from https://cocodataset.org/#download. 
Place the data in a directory structure as the following:
```
${COCO_ROOT}
├── images
|   ├── train2017
|   |   ├── 000000000009.jpg
|   │   ├── ...
|   │   
|   └── val2017
|       ├── 000000000139.jpg
|       ├── ...
|
└── annotations
    ├── person_keypoints_train2017.json
    └── person_keypoints_val2017.json
```

2. Train the model.
```sh
python train.py configs/res50_128x96_d256x3_adam_lr1e-3.yaml --dataset_root ${COCO_ROOT}
```