# two_stream_for_emotion_recognition
Two stream to realize emotion recognition which can be completed as video level. Fork from [two stream pytorch](https://github.com/bryanyzhu/two-stream-pytorch).

# Data preparaton
Use the dataset CK+. In order to generate the optical-flow and rgb images gathered by labels, please refer [prepare-ck-](https://github.com/cMondora/prepare-ck-). This repository is designed to generate the directories like these:
```
|-- rgb                                         |-- flow
|   |-- 1                                       |   |-- 1
|   |   |--S010_004
|   |   |  |-- ***.png
.                                               .
..                                              ..
...                                             ...
|   |-- 6   
|   |   |-- S056
|   |   |--***.png                              |   |   |--***.png
```
# Training
For spatial stream (the last image in every file is used for training which is usually the one that can express the emotion most).
```
[your_path_for_rbg_directory] -m rgb -a rgb_resnet152 --new_length=1 --epochs 250 --lr 0.001 --lr_steps 200 300
```
For temporal stream (The sequential 5 optial-flow images from x_flow and y_flow are used for training, because the least number we have for optial-flow images is 5).
