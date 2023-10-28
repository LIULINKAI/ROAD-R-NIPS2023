import os
import cv2
import csv
import json
import glob
import random
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def remove_all_csv(folder_path):
    print('remove all csv...')
    csv_files = glob.glob(os.path.join(folder_path, '**/*.csv'), recursive=True)
    for file_path in csv_files:
        os.remove(file_path)
        print('removed: ', file_path)

    csv_files = glob.glob(os.path.join(folder_path, '**/*.csv'), recursive=True)
    if len(csv_files) == 0:
        print("Folder and subfolders do not contain any CSV files.")
    else:
        print("Folder and subfolders contain CSV files.")
        print("Number of CSV files found:", len(csv_files))
        input()


def remove_all_local(folder_path):

    print('remove local folder and files...')
    local_folders = glob.glob(os.path.join(folder_path, '**/local'), recursive=True)

    for folder_path in local_folders:
        shutil.rmtree(folder_path)
    
    print('create empty local folder...')
    video_folders = glob.glob(os.path.join(folder_path, '*'))
    for video_folder in video_folders:
        local_folder = os.path.join(video_folder, 'local')
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

def is_part_of_subsets(split_ids, SUBSETS):
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True
    return is_it

def get_label_names(filepath):
    with open(filepath, "r") as f:
        curdict = json.loads(f.read())
    return curdict['agent_names'], curdict['action_names'], curdict['location_names']


def filter_action_labels(action_labels):
    filter_cls_id = [6, 14, 15]
    new_labels = []
    for idx, conf in enumerate(action_labels):
        if idx in filter_cls_id:
            continue
        new_labels.append(conf)
    return new_labels

def filter_action_ids(action_ids):
    new_labels = []
    for id in action_ids:
        action_label = all_action_labels[id]
        if action_label not in action_labels:
            continue
        new_labels.append(action_labels.index(action_label))
    return new_labels

if __name__ == '__main__':
    agent_labels, action_labels, loc_labels = get_label_names("configs/label_names.json")
    # curmode = "valid"
    for curmode in ["train", "valid"]:
        img_folder = os.path.join('base-datasets/act-and-loc-datasets/datasets', curmode)
        gt_file = '../../../road-dataset/road/road_trainval_v1.0.json'
        global_video_folder = '../../../road-dataset/road/rgb-images'
        task1_names = [
            "2014-07-14-14-49-50_stereo_centre_01",
            "2015-02-03-19-43-11_stereo_centre_04",
            "2015-02-24-12-32-19_stereo_centre_04"]
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        
        incomplete_labels = 0

        remove_all_local(img_folder)
        remove_all_csv(img_folder)

        print('Loading json file...')
        with open(gt_file, 'r') as f:
            gt_dict = json.load(f)
            all_action_labels = list(gt_dict['all_action_labels'])

        for video_name, video in tqdm(gt_dict['db'].items(), desc='Processing Video'):
            if curmode is "train" and video_name not in task1_names:
                continue
            if curmode is "valid" and (video_name in task1_names or not is_part_of_subsets(gt_dict["db"][video_name]["split_ids"], ["val_1"])):
                continue
            video_folder = os.path.join(img_folder, video_name)
            for frame_id, data in tqdm(sorted(video['frames'].items(), key=lambda x: int(x[0])), desc=video_name):
                frame_path = os.path.join(global_video_folder, video_name, str(frame_id).zfill(5) + '.jpg')
                img_width, img_height = data['width'], data['height']

                if 'annos' in data:
                    frame_img = cv2.imread(frame_path)
                    if frame_img is None:
                        continue

                    for box_id, annos in data['annos'].items():
                        try:
                            agent_id, tube_uid = annos['agent_ids'][0], annos['tube_uid']
                            if agent_id > 4:
                                agent_id -= 1
                            elif agent_id == 4:
                                print("no class is SMVeh")
                                continue
                            action_ids = annos['action_ids']
                            action_ids = filter_action_ids(action_ids)
                            loc_ids = annos['loc_ids']
                            local_img_path = os.path.join(video_folder, 'local', str(agent_id) + '_' + agent_labels[agent_id], tube_uid)

                            if not os.path.exists(local_img_path):
                                os.makedirs(local_img_path)

                            x1, y1, x2, y2 = annos['box']
                            x1_pixel, y1_pixel, x2_pixel, y2_pixel = round(x1 * img_width) % img_width, round(y1 * img_height) % img_height, round(x2 * img_width) % img_width, round(y2 * img_height) % img_height
                            local_img = frame_img[y1_pixel : y2_pixel, x1_pixel : x2_pixel]

                            if local_img.size != 0:
                                write_img_path = os.path.join(local_img_path, str(frame_id).zfill(5) + '.jpg')
                                if not os.path.exists(write_img_path):
                                    cv2.imwrite(write_img_path, local_img)
                                write_boxes_csv_path = os.path.join(local_img_path, 'boxes.csv')
                                with open(write_boxes_csv_path, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([str(frame_id).zfill(5), x1, y1, x2, y2, agent_id, tube_uid])
                                
                                write_action_csv_path = os.path.join(local_img_path, 'action_label.csv')
                                with open(write_action_csv_path, 'a', newline='') as f:
                                    act_labels = [0 for i in range(len(action_labels))]
                                    for action_id in action_ids:
                                        act_labels[action_id] = 1
                                    writer = csv.writer(f)
                                    writer.writerow(act_labels)

                                write_location_csv_path = os.path.join(local_img_path, 'location_label.csv')
                                with open(write_location_csv_path, 'a', newline='') as f:
                                    location_labels = [0 for i in range(len(loc_labels))]
                                    for loc_id in loc_ids:
                                        location_labels[loc_id] = 1
                                    writer = csv.writer(f)
                                    writer.writerow(location_labels)

                        except IndexError as e:
                            print(e)
                            incomplete_labels += 1

        print('The number of incomplete_labels: ', incomplete_labels)
                    

