
import pickle
import os
import numpy as np
import json
from tqdm import tqdm
def get_req_conf(label_conf):
    new_label_conf = label_conf.copy()
    agent_id = np.argmax(label_conf[:10]) + 1
    down_weight = 1.0 - new_label_conf[agent_id-1]
    for act_id in np.argsort(-label_conf[10:29]):
        cur_act_id = act_id + 10 + 1
        for loc_id in np.argsort(-label_conf[29:]):
            cur_loc_id = loc_id + 29 + 1
            jud = True
            for req_idx in range(len(must_no_cls_ids)):
                if agent_id in must_no_cls_ids[req_idx]:
                    if len(must_true_cls_ids[req_idx]) > 0 and cur_act_id not in must_true_cls_ids[req_idx]:
                        jud = False
                    if cur_act_id in must_no_cls_ids[req_idx]:
                        jud = False
                    if cur_loc_id in must_no_cls_ids[req_idx]:
                        jud = False

                if cur_act_id in must_no_cls_ids[req_idx]:
                    if len(must_true_cls_ids[req_idx]) > 0 and agent_id not in must_true_cls_ids[req_idx]:
                        jud = False
                    if agent_id in must_no_cls_ids[req_idx]:
                        jud = False
                    if agent_id in must_no_cls_ids[req_idx]:
                        jud = False
                
                # if cur_loc_id in must_no_cls_ids[req_idx]:
                #     for loc_id_no in must_no_cls_ids[req_idx]:
                #         if loc_id_no == cur_loc_id or loc_id_no < 30:
                #             continue
                #         new_label_conf[loc_id_no-1] *= down_weight
                if not jud:
                    # new_label_conf[cur_loc_id-1] *= down_weight
                    break
            if not jud:
                new_label_conf[cur_act_id-1] *= down_weight
                # new_label_conf[cur_act_id-1] = 0
            if jud:
                return new_label_conf
    return []
    
if __name__ == '__main__':
    with open(r'submit-task1-v1.pkl', 'rb') as f:
        data = pickle.load(f)
    req_path = "requirements/requirements_dimacs.txt"
    f = open(req_path, 'r')
    must_no_cls_ids = []
    must_true_cls_ids = []
    for idx, line in enumerate(f.readlines()):
        line = line.strip()
        if idx < 1:
            continue
        items = map(int, line.split(' '))
        no_cls_ids = []
        true_cls_ids = []
        for item in items:
            if item < 0:
                no_cls_ids.append(-item)
            elif item > 0:
                true_cls_ids.append(item)
        must_no_cls_ids.append(no_cls_ids)
        must_true_cls_ids.append(true_cls_ids)
    f.close()

    for video_name in data.keys():
        for frame_key in tqdm(data[video_name].keys()):
            new_tube_list = []
            for tube_idx, tube_info in enumerate(data[video_name][frame_key]):
                label_conf = tube_info['labels']
                bbox = tube_info['bbox']
                new_label_conf = get_req_conf(label_conf)
                if len(new_label_conf) == -1:
                    continue
                new_tube_list.append({
                    'bbox':bbox,
                    'labels':np.array(new_label_conf)
                })
            data[video_name][frame_key] = new_tube_list
    
    with open("submit-require-task1.pkl", 'wb') as f:
            pickle.dump(data, f)