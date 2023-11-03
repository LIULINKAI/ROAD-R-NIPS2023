import os
import cv2
import csv
import json
import glob
import random
import shutil
from tqdm import tqdm
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

def is_part_of_subsets(split_ids, SUBSETS):
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True
    return is_it

# 包装成函数:
def get_json_path(json_path):
    with open(json_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_label_names(filepath):
    with open(filepath, "r") as f:
        curdict = json.loads(f.read())
    return curdict['agent_names'], curdict['action_names'], curdict['location_names']

def bbox_to_yolo(_class, x1, y1, x2, y2):
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return str(_class) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h)

if __name__ == '__main__':
    agent_labels, action_labels, loc_labels = get_label_names("configs/label_names.json")
    # curmode = "valid"
    conf_thresh = 0.0 
    img_width = 1280
    img_height = 960
    # pseudo_ratio = 0.2
    # min_pseudo_num = 
    pseudo_label_dir = "output/get_label/get_label_vitcliplarge-base-yolov8-bestacc/2023-11-02-19-05-57"
    task1_labels = ["2014-07-14-14-49-50_stereo_centre_01","2015-02-03-19-43-11_stereo_centre_04","2015-02-24-12-32-19_stereo_centre_04"]
    pseudo_video_names = [
        '2014-06-25-16-45-34_stereo_centre_02',
        '2014-07-14-15-42-55_stereo_centre_03',
        '2014-08-08-13-15-11_stereo_centre_01',
        '2014-08-11-10-59-18_stereo_centre_02',
        '2014-11-14-16-34-33_stereo_centre_06',
        '2014-11-18-13-20-12_stereo_centre_05',
        '2014-11-21-16-07-03_stereo_centre_01',
        '2014-12-09-13-21-02_stereo_centre_01',
        '2015-02-03-08-45-10_stereo_centre_02',
        '2015-02-06-13-57-16_stereo_centre_02',
        '2015-02-13-09-16-26_stereo_centre_05',
        '2015-03-03-11-31-36_stereo_centre_01'
    ]
    print('Loading json file...')
    all_pseudo_dict = {}
    for video_pseudo_file in os.listdir(pseudo_label_dir):
        if ".json" not in video_pseudo_file:
            continue
        video_name = video_pseudo_file.split(".")[0]
        pseudo_dict = get_json_path(os.path.join(pseudo_label_dir, video_pseudo_file))
        all_pseudo_dict[video_name] = pseudo_dict

    curmode = "train"
    img_folder = 'semi-datasets/semi-det-agent/road-r/{}/images'.format(curmode)
    labels_folder = 'semi-datasets/semi-det-agent/road-r/{}/labels'.format(curmode)
    incomplete_labels = 0

    for video_name, video in tqdm(all_pseudo_dict.items(), desc='Processing Video'):
        if curmode is "train" and video_name not in pseudo_video_names:
            continue
        for frame_key, data in tqdm(sorted(video.items(),  key=lambda x: int(x[0].split(".")[0])), desc=video_name):
            frame_id = int(frame_key.split(".")[0]) 
            label_path = os.path.join(labels_folder, "{}_{:05d}.txt".format(video_name, frame_id))
            if len(data) > 0:
                f = open(label_path, "w")
                for tube_id, annos in data.items():
                    try:
                        tube_uid = str(tube_id)
                        label_conf = np.array(annos['labels'])
                        agent_id = np.argmax(label_conf[:len(agent_labels)])
                        if label_conf[agent_id] < conf_thresh:
                            continue

                        x1, y1, x2, y2 = list(annos['bbox'])
                        x1_pixel, y1_pixel, x2_pixel, y2_pixel = round(x1), round(y1), round(x2), round(y2)
                        x1, y1, x2, y2 = x1/img_width, y1/img_height, x2/img_width, y2/img_height
                        linestr = bbox_to_yolo(agent_id, x1, y1, x2, y2) + '\n'
                        f.write(linestr)
                    except IndexError as e:
                        incomplete_labels += 1
                f.close()
        print('The number of incomplete_labels: ', incomplete_labels)
                    

