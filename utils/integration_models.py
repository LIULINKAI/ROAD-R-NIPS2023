import pickle
import os
import numpy as np
import json
from tqdm import tqdm
pkldirs = "utils/test-merge-v2"
pkl_list = os.listdir(pkldirs)
all_data = []
for pklname in pkl_list:
    pklpath = os.path.join(pkldirs, pklname)
    with open(pklpath, 'rb') as f:
        data = pickle.load(f)
        all_data.append(data)
bestlocation = "utils/best_location.pkl"
with open(bestlocation, 'rb') as f:
    locationdata = pickle.load(f)

def compute_iou(box, cls_gt_boxes):
    ious = np.zeros(cls_gt_boxes.shape[0])
    for m in range(cls_gt_boxes.shape[0]):
        gtbox = cls_gt_boxes[m]
        xmin = max(gtbox[0], box[0])
        ymin = max(gtbox[1], box[1])
        xmax = min(gtbox[2], box[2])
        ymax = min(gtbox[3], box[3])
        iw = np.maximum(xmax - xmin, 0.)
        ih = np.maximum(ymax - ymin, 0.)
        if iw > 0 and ih > 0:
            intsc = iw*ih
        else:
            intsc = 0.0
        union = (gtbox[2] - gtbox[0]) * (gtbox[3] - gtbox[1]) + \
            (box[2] - box[0]) * (box[3] - box[1]) - intsc
        ious[m] = intsc/union
    return ious

video_names = all_data[0].keys()
new_data = all_data[0].copy()
iou_thresh = 0.5
location_iou_thresh = 0.5
from tqdm import tqdm
for video_name in video_names:
    for frame_key in tqdm(all_data[0][video_name].keys()):
        merge_data_tube_list = []
        merge_data_tube_num = []
        for data_idx in range(len(all_data)):
            cur_data_tube_list = all_data[data_idx][video_name][frame_key]
            for cur_tube_info in cur_data_tube_list:
                cur_bbox = cur_tube_info['bbox']
                cur_labels = cur_tube_info['labels']
                judmerge = False
                for bbox_idx, tube_info in enumerate(merge_data_tube_list):
                    bbox = tube_info['bbox']
                    labels = tube_info['labels']
                    if compute_iou(cur_bbox, bbox.reshape(1,-1))[0] > iou_thresh:
                        judmerge = True
                        max_conf = max(labels[:10])
                        cur_max_conf = max(cur_labels[:10])
                        if max_conf < cur_max_conf:
                            merge_data_tube_list[bbox_idx]['bbox'] = cur_bbox
                            for i in range(10):
                                merge_data_tube_list[bbox_idx]['labels'][i] = cur_labels[i]
                        
                        cur_max_action_conf = max(cur_labels[10:29])
                        max_action_conf = max(labels[10:29])
                        if max_action_conf < cur_max_action_conf:
                            for i in range(10, 29):
                                merge_data_tube_list[bbox_idx]['labels'][i] = cur_labels[i]

                        # merge_data_tube_list[bbox_idx]['bbox'] = (bbox*merge_data_tube_num[bbox_idx] + cur_bbox) / (merge_data_tube_num[bbox_idx] + 1)
                        # merge_data_tube_list[bbox_idx]['labels'] = (labels*merge_data_tube_num[bbox_idx] + cur_labels) / (merge_data_tube_num[bbox_idx] + 1)
                        # merge_data_tube_num[bbox_idx] += 1
                        break
                if not judmerge:
                    merge_data_tube_list.append(cur_tube_info)
                    merge_data_tube_num.append(1)

        for location_tube_info in locationdata[video_name][frame_key]:
                location_bbox = location_tube_info['bbox']
                location_labels = location_tube_info['labels']
                judmerge = False
                for bbox_idx, tube_info in enumerate(merge_data_tube_list):
                    bbox = tube_info['bbox']
                    labels = tube_info['labels']
                    if compute_iou(location_bbox, bbox.reshape(1,-1))[0] > location_iou_thresh:
                        judmerge = True
                        max_location_loc_conf = max(location_labels[29:])
                        max_loc_conf = max(labels[29:])
                        if max_location_loc_conf > max_loc_conf:
                            for i in range(29,41):
                                merge_data_tube_list[bbox_idx]['labels'][i] = location_labels[i]
                        break
        
        new_data[video_name][frame_key] = merge_data_tube_list

with open("submit-task1.pkl", 'wb') as f:
        pickle.dump(new_data, f)