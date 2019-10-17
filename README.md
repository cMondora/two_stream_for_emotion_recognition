# two_stream_for_emotion_recognition
Two stream to realize emotion recognition which can be completed as video level. Fork from [two stream pytorch](https://github.com/bryanyzhu/two-stream-pytorch).

# Data preparaton
Use the dataset CK+. In order to generate the optical-flow and rgb images gathered by labels, please refer [prepare-ck-](https://github.com/cMondora/prepare-ck-). This repository is designed to generate a directory like this:
```
|-- rgb
|   |-- 1
|   |   |--***.png
...
|   |-- 6
|   |   |--***.png
```
