
## Overview
```
|——base-datasets
|   |——act-and-loc-datasets
|   |   |——datasets
|   |   |   |——train
|   |   |   |——valid
|   |——datasets
|   |   |——road-r
|   |   |   |——train
|   |   |   |——valid
|——semi-datasets
|   |——semi-det-agent
|   |   |——road-r
|   |   |   |——train
|   |   |   |——valid
```


### First-stage: Agent Detection
#### Agent Data preprocessing

`python utils/road2yolo.py`

In the first stage of model training, we use the YOLOv8 algorithm to detect the type and bounding box of the agent in each video frame. Specifically, we use [ultralytics](https://github.com/ultralytics/ultralytics)’ official yolov8l.pt model as the pre-training model. During the training process, we only used the supervision information in the three videos 2014-07-14-14-49-50_stereo_centre_01, 2015-02-03-19-43-11_stereo_centre_04, 2015-02-24-12-32-19_stereo_centre_04, As shown in [main-yolov8.py](./main-yolov8.py). The data preprocessing code is shown in [road2yolo.py](utils/road2yolo.py).

#### Train yolov8l

`python main-yolov8.py`

The input size of the video frame is 1280*960, and the batch-size is set to 4 (See the [code](./main-yolov8.py) for more details).

(Optional) To execute [pseudo_label2yolo.py](utils/pseudo_label2yolo.py) to generate pseudo-label data in yolo format from the model's prediction results, which can further fine-tune the first stage model, but the performance improvement brought about is not significant. We believe there are better semi-supervised options that have not been explored.


### Second-stage: Action and Location Multi-label classification

#### Data preprocessing

In the second stage of model training, we still use the annotation data in the three videos mentioned above (2014-07-14-14-49-50_stereo_centre_01, 2015-02-03-19-43-11_stereo_centre_04, 2015-02-24-12-32-19_stereo_centre_04), and extract target sequence images of different types of agents as training sets.

`python utils/act-and-loc-data_processor.py`

#### Action and Location Classifier
![Alt text](images/image.png)

First, we extract the regional image sequences of agents in the video as input, and use the AIM-Vit-L model proposed in ["AIM: Adapting Image Models for Efficient Video Action Recognition" (ICLR 2023)](https://openreview.net/pdf?id=CIoSZ_HKHS7) as the action and location classifier. We changed the loss function in the training phase to the sigmoid_focal_loss function and trained for 10 epochs.

`python train_stage_two-large.py --dataset_path base-datasets/act-and-loc-datasets/datasets --window_size 4 --head_mode 2d --model vit_clip_pro --use_local`

## Inference

`python inference-stage-two-no-agent-id.py --yolo_path runs/detect/yolov8l_base_1280_batch_4_agent/weights/best.pt --classifier_path runs/exp-stage2-vit_clip-2023-10-21-22_46_31/weight/best_weight.pt  --windows_size 4 --yolo_name base --test_mode test`

Executing the above command can generate a submittable file in the ./output folder. 


[integration_models.py](utils/integration_models.py) integrates multiple prediction results of models trained under different hyperparameter settings, making the final result of the model more stable. [submit_requirement_for_task1.py](utils/submit_requirement_for_task1.py) is used to post-process the prediction results according to the logical constraints in [requirements](requirements/requirements_dimacs.txt).

## Requirement
`
Python 3.8.16,
We achieve equivalent results on two machines: (cuda V11.5, NVIDIA GeForce RTX 3090) and (cuda V10.1, NVIDIA GeForce RTX 2080 Ti).
`

## Acknowledgments

Our code is implemented based on [ROAD-R-2023-Challenge](https://github.com/mihaela-stoian/ROAD-R-2023-Challenge)、[ultralytics](https://github.com/ultralytics/ultralytics)、[adapt-image-models](https://github.com/taoyang1122/adapt-image-models) and [ROADpp_challenge_ICCV2023](https://github.com/ricky-696/ROADpp_challenge_ICCV2023), thanks to these outstanding work.