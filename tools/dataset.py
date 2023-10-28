import os
import glob
from tqdm import tqdm
from collections import deque

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

def bbox_normalized(bbox, img_w, img_h):
    return bbox / np.array([img_w, img_h, img_w, img_h])

class mydataset(nn.Module):
    def __init__(self, args, is_train=True, use_local=True, return_agent=False) -> None:
        self.datapath = os.path.join(args.dataset_path, "train" if is_train else "valid")
        self.global_img_path = args.global_img_path
        self.datatype = args.target
        self.window_size = int(args.window_size)
        self.shape = list(map(int, args.input_shape))
        self.agent_labels = args.agent_labels
        self.action_labels = args.action_labels
        self.loc_labels = args.loc_labels
        self.action_order = [i for i in range(len(self.action_labels))]
        self.loc_order = [i for i in range(len(self.loc_labels))]
        self.use_local = use_local
        self.return_agent = return_agent

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        videos = glob.glob(self.datapath + "/*/")

        self.meta_tube = {} # key for ids value for labels
        for video in videos:
            video_id = video.split('/')[-2]

            all_tubes = glob.glob(video + 'local/*/*/')
            for tube in all_tubes:
                tube_id = tube.split('/')[-2]
                self.meta_tube[tube_id] = {
                    'video_id': video_id,
                    'frame_id': [],
                    'bbox_pos': [],
                    'agent_id': [],
                    'action_labels': [],
                    'loc_labels': []
                }
                if not os.path.exists(tube + 'boxes.csv'):
                    continue

                with open(tube + 'action_label.csv', 'r') as f:
                    action_labels_for_tube = f.readlines()
                with open(tube + 'location_label.csv', 'r') as f:
                    loc_labels_for_tube = f.readlines()
                with open(tube + 'boxes.csv', 'r') as f:
                    boxes_for_tube = f.readlines()

                for idx, labels in enumerate(boxes_for_tube):
                    frame_id, x1, y1, x2, y2, agent_id, _ = labels.split(',')

                    self.meta_tube[tube_id]['frame_id'].append(frame_id + '.jpg')
                    self.meta_tube[tube_id]['bbox_pos'].append(list(map(float, [x1, y1, x2, y2])))
                    self.meta_tube[tube_id]['agent_id'].append(agent_id)
                    self.meta_tube[tube_id]['action_labels'].append(list(map(int, action_labels_for_tube[idx].split(','))))
                    self.meta_tube[tube_id]['loc_labels'].append(list(map(int, loc_labels_for_tube[idx].split(','))))
        
        # self.video_ids = self.meta_data.keys()
        self.all_datas = self.combine_datas(self.meta_tube)

    
    def __getitem__(self, idx):
        window = self.all_datas[idx]

        datas = {'stacked_img': [], 'label': []}

        for frame in window:
            video_id = frame['video_id']
            tube_id = frame['tube_id']
            local_frame_id = frame['local_frame_id']
            global_frame_id = frame['global_frame_id']
            bbox_pos = frame['bbox_pos']
            action_labels = frame['action_labels']
            agent_id = frame['agent_id']
            loc_labels = frame['loc_labels']
            
            global_img = cv2.imread(os.path.join(
                self.global_img_path, video_id, global_frame_id))
            global_img = cv2.resize(cv2.cvtColor(global_img, cv2.COLOR_BGR2RGB), self.shape)

            if self.use_local:
                local_img = cv2.imread(os.path.join(self.datapath, video_id, 'local', 
                    str(agent_id) + '_' + self.agent_labels[int(agent_id)], tube_id, local_frame_id))
                local_img = cv2.resize(cv2.cvtColor(local_img, cv2.COLOR_BGR2RGB), self.shape)
                stack_img = np.concatenate((global_img, local_img), axis=-1)
            else:
                stack_img = global_img

            label_dict = {
                'data_anno': {
                    'video_id': video_id,
                    'tube_id': tube_id,
                    'frame_id': local_frame_id,
                    'agent_id': agent_id + '_{}'.format(self.agent_labels[int(agent_id)]),

                },
                'bbox_pos': bbox_pos,
                'action_label': action_labels,
                'loc_label': loc_labels
            }
            label_dict['action_label'] = torch.FloatTensor(label_dict['action_label'])
            label_dict['loc_label'] = torch.FloatTensor(label_dict['loc_label'])

            datas['stacked_img'].append(self.transform(stack_img))
            datas['label'].append(label_dict)

        data = torch.stack(datas['stacked_img'], dim=0).transpose(0,1)
        if self.datatype == "action":
                label = torch.stack([y["action_label"] for y in datas['label']], dim=0)
                return data, label
        elif self.datatype == "location":
                label = torch.stack([y["loc_label"] for y in datas['label']], dim=0)
                boxes = torch.stack([torch.FloatTensor(y["bbox_pos"]) for y in datas['label']], dim=0)
                return data, label, boxes
        else:
            label = torch.stack([torch.cat([y["action_label"], y["loc_label"]], dim=-1) for y in datas['label']], dim=0)
            boxes = torch.stack([torch.FloatTensor(y["bbox_pos"]) for y in datas['label']], dim=0)
            if self.return_agent:
                agent_ids = torch.LongTensor([int(y["data_anno"]['agent_id'].split("_")[0]) for y in datas['label']])
                agent_ids = agent_ids.view(self.window_size, -1)
                # agent_id = torch.stack(agent_ids, dim=0)
                return data, label, boxes, agent_ids
            return data, label, boxes

    
    def __len__(self):
        return len(self.all_datas)


    def combine_datas(self, meta_tube):
        all_datas = []

        for key in meta_tube.keys():
            tubes = meta_tube[key]

            window_queue = deque(maxlen=self.window_size)
            for i in range(len(tubes['frame_id'])):
                window_queue.append({
                    'video_id': tubes['video_id'],
                    'tube_id': key,
                    'local_frame_id': tubes['frame_id'][i],
                    'global_frame_id': tubes['frame_id'][i],
                    'bbox_pos': tubes['bbox_pos'][i],
                    'agent_id': tubes['agent_id'][i],
                    'action_labels': tubes['action_labels'][i],
                    'loc_labels': tubes['loc_labels'][i]
                })

                if len(window_queue) == self.window_size:
                    all_datas.append(window_queue.copy())
            if len(window_queue) > 0 and len(window_queue) < self.window_size:
                num = self.window_size - len(window_queue)
                for i in range(num):
                    window_queue.append({
                        'video_id': tubes['video_id'],
                        'tube_id': key,
                        'local_frame_id': tubes['frame_id'][-1],
                        'global_frame_id': tubes['frame_id'][-1],
                        'bbox_pos': tubes['bbox_pos'][-1],
                        'agent_id': tubes['agent_id'][-1],
                        'action_labels': tubes['action_labels'][-1],
                        'loc_labels': tubes['loc_labels'][-1]
                    })
                all_datas.append(window_queue.copy())
        return all_datas


class myTestdataset(nn.Module):
    def __init__(self, args, is_train=False) -> None:
        self.datapath = os.path.join(args.dataset_path, "train" if is_train else "valid")
        self.global_img_path = args.global_img_path
        self.datatype = args.target
        self.window_size = int(args.windows_size)
        self.shape = list(map(int, args.input_shape))
        self.agent_labels = args.agent_labels
        self.action_labels = args.action_labels
        self.loc_labels = args.loc_labels
        self.action_order = [i for i in range(len(self.action_labels))]
        self.loc_order = [i for i in range(len(self.loc_labels))]
        h, w = args.video_shape

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        videos = glob.glob(self.datapath + "/*/")

        self.meta_tube = {} # key for ids value for labels
        for video in videos:
            video_id = video.split('/')[-2]

            all_tubes = glob.glob(video + 'local/*/*/')
            for tube in all_tubes:
                tube_id = tube.split('/')[-2]
                self.meta_tube[tube_id] = {
                    'video_id': video_id,
                    'frame_id': [],
                    'boxes': [],
                    'agent_id': [],
                    'action_labels': [],
                    'loc_labels': [],
                    'stack_imgs':[]
                }

                with open(tube + 'action_label.csv', 'r') as f:
                    action_labels_for_tube = f.readlines()
                with open(tube + 'location_label.csv', 'r') as f:
                    loc_labels_for_tube = f.readlines()
                with open(tube + 'boxes.csv', 'r') as f:
                    boxes_for_tube = f.readlines()

                for idx, labels in enumerate(boxes_for_tube):
                    frame_id, x1, y1, x2, y2, agent_id, _ = labels.split(',')
                    boxes = list(map(float, [x1, y1, x2, y2]))
                    boxes[0], boxes[2] = boxes[0]*w, boxes[2]*w
                    boxes[1], boxes[3] = boxes[1]*h, boxes[3]*h

                    self.meta_tube[tube_id]['frame_id'].append(frame_id + '.jpg')
                    self.meta_tube[tube_id]['boxes'].append(boxes)
                    self.meta_tube[tube_id]['agent_id'].append(agent_id)
                    self.meta_tube[tube_id]['action_labels'].append(list(map(int, action_labels_for_tube[idx].split(','))))
                    self.meta_tube[tube_id]['loc_labels'].append(list(map(int, loc_labels_for_tube[idx].split(','))))

                    local_frame_id = frame_id + '.jpg'
                    global_frame_id = frame_id + '.jpg'
                    local_img_path = os.path.join(self.datapath, video_id, 'local', str(agent_id) + '_' + self.agent_labels[int(agent_id)], tube_id, local_frame_id)
                    global_img_path = os.path.join(self.global_img_path, video_id, global_frame_id)
                    self.meta_tube[tube_id]['stack_imgs'].append([global_img_path, local_img_path])

        self.all_datas = list(self.meta_tube.keys())
    def __getitem__(self, idx):
        tube_id = self.all_datas[idx]
        for idx, (global_img_path, local_img_path) in enumerate(self.meta_tube[tube_id]['stack_imgs']):
            local_img = cv2.imread(local_img_path)
            global_img = cv2.imread(global_img_path)
            
            local_img = cv2.resize(cv2.cvtColor(local_img, cv2.COLOR_BGR2RGB), self.shape)
            global_img = cv2.resize(cv2.cvtColor(global_img, cv2.COLOR_BGR2RGB), self.shape)

            stack_img = np.concatenate((global_img, local_img), axis=-1)
            self.meta_tube[tube_id]['stack_imgs'][idx] = stack_img
        return self.meta_tube[tube_id]
    
    def __len__(self):
        return len(self.all_datas)
    
    
class Tracklet_Dataset(nn.Module):
    def __init__(self, mode, tracklet, args, bbox=None):
        self.mode = mode
        self.bbox = []
        self.img_w = args.submit_shape[1]
        self.img_h = args.submit_shape[0]
        self.to_tensor = transforms.ToTensor()

        self.windows = []
        windows_size = args.windows_size
        windows_deque = deque(maxlen=windows_size)
        if self.mode != 'action':
            boxes_deque = deque(maxlen=windows_size)

        for idx, t in enumerate(tracklet):
            windows_deque.append(self.to_tensor(t))
            if self.mode != 'action':
                boxes_deque.append(torch.tensor(bbox_normalized(bbox[idx], self.img_w, self.img_h), dtype=torch.float32))
            if len(windows_deque) == windows_size:
                stacked_img = torch.stack(list(windows_deque), dim=0).transpose(0,1)
                if self.mode != 'action':
                    box = torch.stack(list(boxes_deque), dim=0)
                    self.bbox.append(box)
                self.windows.append(stacked_img)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if self.mode != 'action':
            return self.windows[idx], self.bbox[idx]
        else:
            return self.windows[idx]

class trainvalTracklet_Dataset(nn.Module):
    def __init__(self, mode, tracklet, args, bbox=None, labels=None):
        self.mode = mode
        self.bbox = []
        self.img_w = args.submit_shape[1]
        self.img_h = args.submit_shape[0]
        self.to_tensor = transforms.ToTensor()

        self.windows = []
        windows_size = args.windows_size
        windows_deque = deque(maxlen=windows_size)
        if self.mode != 'action':
            boxes_deque = deque(maxlen=windows_size)

        for idx, t in enumerate(tracklet):
            windows_deque.append(self.to_tensor(t))
            if self.mode != 'action':
                boxes_deque.append(torch.tensor(bbox_normalized(bbox[idx], self.img_w, self.img_h), dtype=torch.float32))
            if len(windows_deque) == windows_size:
                stacked_img = torch.stack(list(windows_deque), dim=0).transpose(0,1)
                if self.mode != 'action':
                    box = torch.stack(list(boxes_deque), dim=0)
                    self.bbox.append(box)
                self.windows.append(stacked_img)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if self.mode != 'action':
            return self.windows[idx], self.bbox[idx]
        else:
            return self.windows[idx]
