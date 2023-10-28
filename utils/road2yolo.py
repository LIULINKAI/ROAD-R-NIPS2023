import os
import re
import cv2
import glob
import json
import random
import shutil
from tqdm import tqdm
def is_part_of_subsets(split_ids, SUBSETS):
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True
    return is_it

def debug_draw(img, b, filename, w, h):
    x1, y1, x2, y2 = b
    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    cv2.rectangle(img, (x1, y1), (x2, y2),(0, 0, 255), 3)
    cv2.imwrite(filename, img)


def bbox_to_yolo(_class, x1, y1, x2, y2):
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return str(_class) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h)


def img_to_yolo(video_folder, cur_folder, subset="train_1"):
    # 取得所有影片資料夾的名稱
    video_folders = sorted(os.listdir(video_folder))

    # 迭代每個影片資料夾
    for video_folder_name in tqdm(video_folders):
        if video_folder_name not in gt_dict["db"].keys():
            continue
        if "train" in subset:
            if video_folder_name not in task1_labels or not is_part_of_subsets(gt_dict["db"][video_folder_name]["split_ids"], [subset]):
                continue
        elif "val" in subset:
            if video_folder_name in task1_labels or not is_part_of_subsets(gt_dict["db"][video_folder_name]["split_ids"], [subset]):
                continue

        video_path = os.path.join(video_folder, video_folder_name)
        
        # 檢查路徑是否為資料夾
        if os.path.isdir(video_path):
            # 取得影片名稱
            video_name = video_folder_name
            
            # 取得影像檔案清單
            image_files = os.listdir(video_path)
            
            # 迭代每個影像檔案
            for image_file in image_files:
                image_path = os.path.join(video_path, image_file)
                
                # 檢查檔案是否為影像
                if os.path.isfile(image_path) and image_file.endswith(".jpg"):
                    # 取得frame_id
                    frame_id = image_file.split(".")[0]
                    
                    # 設定目標檔案路徑
                    target_path = os.path.join(cur_folder, f"{video_name}_{frame_id}.jpg")
                    
                    # 複製影像檔案到目標資料夾並重新命名
                    shutil.copy(image_path, target_path)
                    # print(f"copy {image_path} to {target_path}")


def fix_agent_labels(labels_id):
    if labels_id > 4:
        labels_id -= 1
    elif labels_id == 4:
        labels_id = -1
    return labels_id

def gt_to_yolo(gt_file, subset="train_1"):

    for video_name, video in tqdm(gt_dict['db'].items()):
        if "train" in subset:
            save_path = os.path.join(train_folder, 'labels')
            if video_name not in task1_labels or not is_part_of_subsets(gt_dict["db"][video_name]["split_ids"], [subset]):
                continue
        else:
            save_path = os.path.join(val_folder, 'labels')
            if video_name in task1_labels or not is_part_of_subsets(gt_dict["db"][video_name]["split_ids"], [subset]):
                continue
        for frame_id, data in video['frames'].items():
            file_name = str(video_name) + '_' + str(frame_id).zfill(5)
            img_width, img_height = data['width'], data['height']

            # img = cv2.imread(os.path.join(train_folder, file_name + '.jpg'))

            save_file = os.path.join(save_path, file_name + '.txt')
            # print(f'writing {save_file}')

            # bbox format: x1, y1, x2, y2
            if 'annos' in data:
                for box_id, annos in data['annos'].items():
                    agent_label = annos['agent_ids'][0]
                    agent_label = fix_agent_labels(int(agent_label))
                    if agent_label == -1:
                        continue
                    if len(annos['agent_ids']) > 1:
                        print("{}中agent数量大于1".format(save_file))
                    x1, y1, x2, y2 = annos['box']
                    yolo_bbox_label = bbox_to_yolo(agent_label, x1, y1, x2, y2)

                    if not os.path.exists(save_file):
                        with open(save_file, 'w') as f:
                            f.write(yolo_bbox_label)
                    else:
                        with open(save_file, 'a') as f:
                            f.write('\n' + yolo_bbox_label)
                


def cut_train_valid(train_folder, val_folder, mode='video', ratio=0.9):
    train_images_folder = os.path.join(train_folder, 'images')
    train_labels_folder = os.path.join(train_folder, 'labels')
    valid_images_folder = os.path.join(val_folder, 'images')
    valid_labels_folder = os.path.join(val_folder, 'labels')

    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(valid_images_folder, exist_ok=True)
    os.makedirs(valid_labels_folder, exist_ok=True)

    train_files = os.listdir(train_images_folder)

    if mode == 'video':
        video_names = set()
        for file in train_files:
            video_name = file.split('_')[1]
            video_names.add(video_name)

        num_videos = len(video_names)
        num_valid_videos = int(num_videos * (1 - ratio))
        valid_video_names = set(random.sample(video_names, num_valid_videos))

        for file in train_files:
            video_name = file.split('_')[1]

            if video_name in valid_video_names:
                shutil.move(os.path.join(train_images_folder, file), os.path.join(valid_images_folder, file))
                shutil.move(os.path.join(train_labels_folder, file.replace('.jpg', '.txt')), os.path.join(valid_labels_folder, file.replace('.jpg', '.txt')))
                print(f'move {video_name}')
    
    elif mode == 'random':
        pass


def check_and_delete_files(cur_folder):
    labels_folder = os.path.join(cur_folder, 'labels')
    images_folder = os.path.join(cur_folder, 'images')
    unlabel_images_folder = os.path.join(cur_folder, 'unlabel_images')
    if not os.path.exists(unlabel_images_folder):
        os.makedirs(unlabel_images_folder)

    img_files = os.listdir(images_folder)

    for img in tqdm(img_files):
        file_name = img.split('.')[0]
        label_path = os.path.join(labels_folder, file_name + '.txt')

        if not os.path.exists(label_path):
            image_file_path = os.path.join(images_folder, img)
            dst_file_path = os.path.join(unlabel_images_folder, img)
            if os.path.exists(dst_file_path):
                continue
            shutil.move(image_file_path, dst_file_path)
            # print(f"Deleted file: {image_file_path}")


def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    img_path = os.path.join(folder_path, 'images')
    label_path = os.path.join(folder_path, 'labels')
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)


def cut_two_branch_yolo(ori_folder, new_folder, cls):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # 遍历train和valid子文件夹
    for split_folder in ["train", "valid"]:
        ori_split_path = os.path.join(ori_folder, split_folder)
        new_split_path = os.path.join(new_folder, split_folder)

        # 遍历images和labels子文件夹
        for data_folder in ["images", "labels"]:
            new_data_path = os.path.join(new_split_path, data_folder)
            if not os.path.exists(new_data_path):
                os.makedirs(new_data_path)

        # 遍历labels文件夹中的txt文件
        label_files = glob.glob(os.path.join(ori_split_path, "labels", "*.txt"))
        for label_file in tqdm(label_files):
            
            txt_file_name = label_file.split('/')[-1]
            img_file_name = txt_file_name.split('.')[0] + '.jpg'

            # 读取txt文件内容
            with open(label_file, 'r') as f:
                lines = f.readlines()

            copy_data = []
            # 遍历每一行txt文件
            for line in lines:
                label = int(line.strip().split()[0])

                if label in cls:
                    new_label = cls.index(label)
                    new_line = re.sub(r'^\d+\b', str(new_label), line)
                    copy_data.append(new_line)

            if len(copy_data) > 0:
                new_txt_path = os.path.join(new_split_path, "labels", txt_file_name)
                with open(new_txt_path, 'w') as f:
                    for data in copy_data:
                        f.write(data)

                # 构建对应的jpg文件名
                ori_jpg_path = os.path.join(ori_split_path, "images", img_file_name)
                new_jpg_path = os.path.join(new_split_path, "images", img_file_name)

                # 复制jpg文件
                shutil.copy(ori_jpg_path, new_jpg_path)


def delete_files(folder):
    # 检查文件夹是否存在
    if os.path.exists(folder):
        # 获取文件夹中所有文件的列表
        files = os.listdir(folder)
        
        # 遍历文件列表并删除每个文件
        for file in files:
            file_path = os.path.join(folder, file)
            shutil.rmtree(file_path)

        print(f"已删除文件夹 {folder} 中的所有文件")
    else:
        print(f"文件夹 {folder} 不存在")

if __name__ == '__main__':
    # create yolo datasets
    img_folder = '../../../road-dataset/road/rgb-images'
    train_folder = 'base-datasets/datasets/road-r/train'
    val_folder = 'base-datasets/datasets/road-r/valid'
    # test_floder = 'base-datasets/datasets/road-r/test'
    gt_file = '../../../road-dataset/road/road_trainval_v1.0.json'
    task1_labels = [
        "2014-07-14-14-49-50_stereo_centre_01",
        "2015-02-03-19-43-11_stereo_centre_04",
        "2015-02-24-12-32-19_stereo_centre_04"
        ]
    f = open(gt_file, "r")
    gt_dict = json.loads(f.read())
    f.close()
    # TODO 创建相关文件夹
    create_folder(train_folder)
    create_folder(val_folder)
    # create_folder(test_floder)

    # TODO 生成训练集和验证集
    img_to_yolo(img_folder, os.path.join(train_folder, 'images'), subset="train_1")
    gt_to_yolo(gt_file, subset="train_1")
    check_and_delete_files(train_folder)

    img_to_yolo(img_folder, os.path.join(val_folder, 'images'), subset="val_1")
    gt_to_yolo(gt_file, subset="val_1")
    check_and_delete_files(val_folder)
    
