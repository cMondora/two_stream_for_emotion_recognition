# two_stream_for_emotion_recognition
Two stream to realize emotion recognition which can be completed as video level. Fork from [two stream pytorch](https://github.com/bryanyzhu/two-stream-pytorch).

## Data preparaton
### Raw dataset -> rgb & flow images
Use the dataset CK+. In order to generate the optical-flow and rgb images gathered by labels, please refer [prepare-ck-](https://github.com/cMondora/prepare-ck-). This repository is designed to generate the directories like these:
```
|-- rgb                                         |-- flow
|   |-- 1                                       |   |-- 1
|   |   |-- S010_004                            |   |   |-- S010_004
|   |   |   |-- ***.png                         |   |   |   |-- x_flow
.                                               |   |   |   |   |-- ***.png  
..                                              |   |   |   |-- x_flow
...                                             |   |   |   |   |-- ***.png  
|   |-- 6   
|   |   |-- S056_002                            .
|   |   |   | ***.png                           ..
```

### Split training and test set
Run split.py to split training and test set.
You can find the txts in ./datasets/settings/ck/
```
train_flow_split.txt
val_flow_split.txt
train_rgb_split.txt
val_rgb_split.txt
```

## Training
Run main_single_gpu.py
For spatial stream (the last image in every file is used for training which is usually the one that can express the emotion most).
```
[your_path_for_rbg_directory] -m rgb -a rgb_resnet152 --new_length=1 --epochs 250 --lr 0.001 --lr_steps 100 200
```
For temporal stream (The sequential 5 optial-flow images from x_flow and y_flow are used for training, because the least number we have for optial-flow images is 5).  As a result the input for temporal stream is the 10-channel image.
```
[your_path_for_flow_directory] -m flow -a flow_resnet152 --new_length=5 --epochs 250 --lr 0.001 --lr_steps 200 300
```

## Testing
Run spatial_demo.py and temporal_demo.py for testing.

 network        | top1   |
----------------|:------:|
Spatial stream  | 93.10% | 
Temporal stream | 92.86% | 

