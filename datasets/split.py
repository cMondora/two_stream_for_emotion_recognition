
import argparse
import os
import glob
import random
import fnmatch
import numpy as np
import shutil

def parse_directory(path,modality):
    """
    Parse directories holding extracted frames from standard benchmarks
    """

    def count_files(directory):
        lst = os.listdir(directory)
        cnt = len(fnmatch.filter(lst, '*.png'))+len(fnmatch.filter(lst, '*.jpg'))
        return cnt

    print('parse frames under folder {}'.format(path))
    label_file=glob.glob(os.path.join(path,'*'))
    rgb_counts = {}
    for file in label_file:
        frame_folders=glob.glob(os.path.join(file,'*'))
        for i, f in enumerate(frame_folders):
            k = f.split('/')[-1]
            if modality=='rgb':
                cnt = count_files(f)
            elif modality=='flow':
                cnt = count_files(os.path.join(f,'x_flow'))
            rgb_counts[k] = cnt

    print('frame folder analysis done')
    return rgb_counts


def build_split_list(split_tuple, frame_info, modality,shuffle=False):

    def build_set_list(set_list):
        rgb_list, flow_list = list(), list()
        for item in set_list:
            if modality=='rgb':
                rgb_cnt = frame_info[item[0].split('/')[-1]]
            elif modality=='flow':
                rgb_cnt = frame_info[item[0].split('/')[-2]]
            else:
                print("No such modality. Only rgb and flow supported.")
            rgb_list.append('{} {} {}\n'.format(item[0], rgb_cnt, item[1]))
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list

    train_rgb_list = build_set_list(split_tuple[0])
    test_rgb_list = build_set_list(split_tuple[1])
    return (train_rgb_list, test_rgb_list)


def parse_ck_splits(frame_path,split_rate,modality):
    class_mapping = {'angry':1,'contempt':2,'disgust':3,'fear':4,'happy':5,'sadness':6,'surprise':7}
    train_list=[]
    test_list=[]

    if modality=='rgb':
        for label in os.listdir(frame_path):
            la_path=os.path.join(frame_path,label)
            files_list=os.listdir(la_path)

            shuffle_indexes = np.random.permutation(len(files_list))
            test_size = int(len(files_list) * split_rate)
            for i in range(len(files_list)):
                dx = shuffle_indexes[i]
                if i<test_size:
                    test_list.append((os.path.join(la_path, files_list[dx]), label))
                else:
                    train_list.append((os.path.join(la_path, files_list[dx]), label))
    elif modality=='flow':
        for label in os.listdir(frame_path):
            la_path = os.path.join(frame_path, label)
            files_list = os.listdir(la_path)

            shuffle_indexes = np.random.permutation(len(files_list))
            test_size = int(len(files_list) * split_rate)
            for i in range(len(files_list)):
                dx = shuffle_indexes[i]
                if i < test_size:
                    test_list.append((os.path.join(la_path, files_list[dx],'x_flow'), label))
                else:
                    train_list.append((os.path.join(la_path, files_list[dx],'x_flow'), label))
    else:
        print("No such modality. Only rgb and flow supported.")


    splits = []
    splits.append((train_list, test_list))
    return splits[0]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',type=str,default='ck')
    parser.add_argument('--frame_path', type=str, default='/home/wxc/basedata/expression_data/ck+_pre',
                        help="root directory holding the frames")
    parser.add_argument('--out_list_path', type=str, default='./datasets/settings')
    parser.add_argument('--modality',default='rgb',choices=["rgb", "flow"],help='modality: rgb | flow')
    parser.add_argument('--split_rate',type=float,default=0.1)
    parser.add_argument('--shuffle', action='store_true', default=False)

    args = parser.parse_args()

    frame_path = args.frame_path
    dataset=args.dataset
    out_path = args.out_list_path
    shuffle = args.shuffle
    split_rate=args.split_rate
    modality=args.modality

    save_path = os.path.join(out_path,dataset)
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    #     print("creating folder: "+save_path)
    # os.mkdir(save_path)

    # operation
    print('processing dataset {}'.format(dataset))
    if dataset=='ck':
        split_tp = parse_ck_splits(frame_path,split_rate,modality)
    f_info = parse_directory(frame_path,modality)
    a=set(split_tp[0])
    print(len(a))
    print('writing list files for training/testing')
    lists = build_split_list(split_tp,f_info, modality,shuffle)
    open(os.path.join(save_path, 'train_{}_split.txt'.format(modality)), 'w').writelines(lists[0])
    open(os.path.join(save_path, 'val_{}_split.txt'.format(modality)), 'w').writelines(lists[1])


