#!/usr/bin/env python

import os, sys
import collections
import numpy as np
import cv2
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from VideoSpatialPrediction import VideoSpatialPrediction
import matplotlib.pyplot as plt

sys.path.insert(0, "../../")
import models



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z


def save_fig(pred,clip_path):
    fig_name=clip_path.split('/')[-1]
    # x=np.linspace(1,len(pred),len(pred))
    x = ['angry','contempt','disgust','fear','happy','sadness','surprise']
    plt.bar(x, pred[1:])
    plt.savefig('/home/wxc/.pyenv/versions/3.5.5/envs/two_stream/test/{}.jpg'.format(fig_name))
    plt.clf()

def main():

    # model_path = '../../checkpoints/rgb_model_best.pth.tar'
    model_path = '../../checkpoints/225_rgb_checkpoint.pth.tar'
    # data_dir = "~/basedata/expression_data/ck+_pre"
    start_frame = 0
    num_categories = 8

    model_start_time = time.time()
    params = torch.load(model_path)

    spatial_net = models.rgb_resnet152(pretrained=False, num_classes=8)
    spatial_net.load_state_dict(params['state_dict'])
    spatial_net.cuda()
    spatial_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))

    val_file = "/home/wxc/.pyenv/versions/3.5.5/envs/two_stream/datasets/settings/ck/val_rgb_split.txt"
    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    line_id = 1
    match_count = 0
    result_list = []
    for line in val_list:
        line_info = line.split(" ")
        clip_path = line_info[0]
        input_video_frames=int(line_info[1])
        input_video_label = int(line_info[2])

        spatial_prediction = VideoSpatialPrediction(
                clip_path,
                spatial_net,
                num_categories,
                start_frame)

        avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
        # print(avg_spatial_pred_fc8.shape)
        result_list.append(avg_spatial_pred_fc8)
        avg_spatial_pred = softmax(avg_spatial_pred_fc8)

        save_fig(avg_spatial_pred,clip_path)

        pred_index = np.argmax(avg_spatial_pred_fc8)
        print("Sample %d/%d: number of frames:%d, GT: %d, Prediction: %d, " % (line_id, len(val_list),input_video_frames, input_video_label, pred_index)+clip_path)

        if pred_index == input_video_label:
            match_count += 1
        line_id += 1

    print(match_count)
    print(len(val_list))
    print("Accuracy is %4.4f" % (float(match_count)/len(val_list)))
    np.save("ucf101_s1_rgb_resnet152.npy", np.array(result_list))

if __name__ == "__main__":
    main()




    # # spatial net prediction
    # class_list = os.listdir(data_dir)
    # class_list.sort()
    # print(class_list)

    # class_index = 0
    # match_count = 0
    # total_clip = 1
    # result_list = []

    # for each_class in class_list:
    #     class_path = os.path.join(data_dir, each_class)

    #     clip_list = os.listdir(class_path)
    #     clip_list.sort()

    #     for each_clip in clip_list:
            # clip_path = os.path.join(class_path, each_clip)
            # spatial_prediction = VideoSpatialPrediction(
            #         clip_path,
            #         spatial_net,
            #         num_categories,
            #         start_frame)

            # avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
            # # print(avg_spatial_pred_fc8.shape)
            # result_list.append(avg_spatial_pred_fc8)
            # # avg_spatial_pred = softmax(avg_spatial_pred_fc8)

            # pred_index = np.argmax(avg_spatial_pred_fc8)101
            # print("GT: %d, Prediction: %d" % (class_index, pred_index))

            # if pred_index == class_index:
            #     match_count += 1
#             total_clip += 1

#         class_index += 1

#     print("Accuracy is %4.4f" % (float(match_count)/total_clip))
#     np.save("ucf101_split1_resnet_rgb.npy", np.array(result_list))

# if __name__ == "__main__":
#     main()
