import os
import cv2
import glob
import torch
import pickle
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import datetime
from tools.dataset import Tracklet_Dataset
from tools.linear_interpolation import tube_interpolation
from tools.tube_processing import tube_change_axis, action_tube_padding, stack_imgs_padding, stack_boxes_padding
import argparse
import json

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='Track2', help='detect mode, only accept Track1 or Track2')
    parser.add_argument('--test_mode', type=str, default='test', help='test mode, only accept test or val_1')
    parser.add_argument('--video_path', type=str, default='~/projects/road-r/road-dataset/road/videos', help='video path')
    # parser.add_argument('--yolo_path', type=str, default='~/projects/road-r/yolov8/road-r-nips2023-yolov8/runs/detect/semi-yolov8l_base_1280_batch_4_agent_10classes_pretrained_11classbestmodel/weights/best.pt', help='yolo path')
    parser.add_argument('--yolo_path', type=str, default='~/projects/road-r/yolov8/road-r-2023-yolov8/runs/detect/yolov8l_base_1280_batch_4_agent/weights/best.pt', help='yolo path')
    parser.add_argument('--label_names_file', default="configs/label_names.json", help='path of label names')

    parser.add_argument('--filter_agent', action='store_true', default=False, help='used two branch YOLO')
    parser.add_argument('--filter_action', action='store_true', default=False, help='used two branch YOLO')

    parser.add_argument('--two_branch', action='store_true', default=False, help='used two branch YOLO')
    parser.add_argument('--resume_track1', type=bool, default=True, help='save submit file')
    parser.add_argument('--save_track1_dir', type=str, default='./output-video-tracks', help='output Dir')

    parser.add_argument('--yolo_name', type=str, default='semi', help='semi or base')

    parser.add_argument('--major_path', type=str, default='~/projects/road-r/yolov8/road-r-nips2023-yolov8/runs/detect/yolov8l_major_v2_1280_batch_4_agent2/weights/best.pt', help='major_yolo path')
    parser.add_argument('--rare_path', type=str, default='~/projects/road-r/yolov8/road-r-nips2023-yolov8/runs/detect/yolov8l_rare_v2_1280_batch_4_agent/weights/best.pt', help='rare_yolo path')

    parser.add_argument('--devices', type=str, default='2', help='gpu id')

    parser.add_argument('--imgsz', type=tuple, default=(960, 1280), help='yolo input size')
    parser.add_argument('--video_shape', type=tuple, default=(960, 1280), help='original video resolution')
    parser.add_argument('--submit_shape', type=tuple, default=(960, 1280), help='final submit shape')

    parser.add_argument('--pkl_name', type=str, default=None, help='submit file name(*.pkl)')
    parser.add_argument('--save_res', type=bool, default=True, help='save submit file')
    parser.add_argument('--outputdir', type=str, default='./output', help='output Dir')
    parser.add_argument('--resume', type=str, default=None, help='output Dir')

    # track2
    # test mAP=24.03 (2D head best loss weights)
    parser.add_argument('--classifier_path', type=str, default='~/projects/road-r/yolov8/road-r-nips2023-yolov8/runs/vitlargeCLIP-winsz4-2dhead-use-local2023-10-25-17_12_37/weight/best_weight.pt', help='action and location detector_path')
    
    parser.add_argument('--t2_input_shape', type=tuple, default=(224, 224), help='t2_input_shape')
    parser.add_argument('--windows_size', type=int, default=4, help='sliding windows shape')
    

    opt = parser.parse_args()
    return opt

def out_of_range(x, y, max_x, max_y):
    x = min(max(x, 0), max_x)
    y = min(max(y, 0), max_y)
    return x, y

def filter_agent_labels(labels_id):
    if labels_id == 4:
        labels_id = -1
    elif labels_id > 4:
        labels_id -= 1
    return labels_id

def filter_agent_conf(label_conf):
    new_conf = []
    for idx, conf in enumerate(label_conf):
        if idx == 4:
            continue
        new_conf.append(conf)
    return new_conf

def make_tube(args):
    tracklet = {}
    stack_imgs = {} 
    frame_num = 0
    
    # Tracker.boxes.data(Tensor): x1, y1, x2, y2, track_id, conf, label_id
    for t, cls_confs in args.tracker:
        frame_num += 1
        
        if t.boxes.is_track:
            frame_img = t.orig_img
            global_img = cv2.resize(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), args.t2_input_shape)

            for box_id, b in enumerate(t.boxes.data):
                x1, y1, x2, y2, track_id, conf, label_id = b
                
                # Convert tensor values to Python scalars
                x1, y1, x2, y2, track_id, conf, label_id = (
                    x1.item(), y1.item(), x2.item(), y2.item(),
                    int(track_id.item()), conf.item(), int(label_id.item())
                )
                track_id = int(track_id)
                track_id += args.start_track_id

                # cur_cls_conf = [0.0 for i in range(10)]
                cur_cls_conf = list(cls_confs[box_id].cpu().numpy())
                cur_cls_conf = filter_agent_conf(cur_cls_conf) if not args.two_branch and args.filter_agent else cur_cls_conf
                label_id = filter_agent_labels(label_id) if not args.two_branch and args.filter_agent else label_id

                x1, y1 = out_of_range(x1, y1, t.orig_shape[1], t.orig_shape[0])
                x2, y2 = out_of_range(x2, y2, t.orig_shape[1], t.orig_shape[0])

                if args.mode == 'Track2': # Fusion of region level images and global image information
                    local_img = frame_img[int(y1) : int(y2), int(x1) : int(x2)]
                    local_img = cv2.resize(cv2.cvtColor(local_img, cv2.COLOR_BGR2RGB), args.t2_input_shape)
                    stack_img = np.concatenate((global_img, local_img), axis=-1)

                if track_id not in tracklet:
                    # agent
                    tracklet[track_id] = {
                        'label_id': int(label_id),
                        'scores': np.array([conf]),
                        'boxes': np.array([[x1, y1, x2, y2]]),
                        'score': 0.0,
                        'frames': np.array([frame_num]),
                        # 'cls_conf': np.array([cur_cls_conf]),
                        'cls_conf': [cur_cls_conf],
                        "track_id":int(track_id)
                    }

                    # event
                    if args.mode == 'Track2':
                        stack_imgs[track_id] = [stack_img]
                else:
                    # agent
                    tracklet[track_id]['scores'] = np.append(tracklet[track_id]['scores'], conf)
                    tracklet[track_id]['boxes'] = np.append(tracklet[track_id]['boxes'], [[x1, y1, x2, y2]], axis=0)
                    tracklet[track_id]['frames'] = np.append(tracklet[track_id]['frames'], frame_num)

                    tracklet[track_id]['cls_conf'].append(cur_cls_conf)

                    # event
                    if args.mode == 'Track2':
                        stack_imgs[track_id].append(stack_img)

                frame_key = "{:05d}.jpg".format(frame_num)
                if frame_key not in submit_pkl_json[args.video_name].keys() or len(submit_pkl_json[args.video_name][frame_key]) == 0:
                    submit_pkl_json[args.video_name][frame_key] = {}
                cur_track_info = {
                    "bbox":np.array([x1, y1, x2, y2]),
                    "labels":list(cur_cls_conf)
                }
                submit_pkl_json[args.video_name][frame_key][int(track_id)] = cur_track_info
        else:
            frame_key = "{:05d}.jpg".format(frame_num)
            if frame_key not in submit_pkl_json[args.video_name].keys() or len(submit_pkl_json[args.video_name][frame_key]) == 0:
                    submit_pkl_json[args.video_name][frame_key] = []

    event_list = []
    for tube_id, tube_data in tracklet.items():
        # agent
        if args.mode == 'Track1': # if do interpolation in T2, len(tube_data['frames']) != len(stack_imgs[tube_id])
            tube_data = tube_interpolation(tube_data)
            
        tube_data = tube_change_axis(tube_data, args.video_shape, args.submit_shape) # change axis to submit_shape
        tube_data['score'] = np.mean(tube_data['scores'])
        # event
        if args.mode == 'Track2':
            tube_data['stack_imgs'] = stack_imgs[tube_id]
            event_list.append(tube_data)

    if args.two_branch:
        return event_list, frame_num
    else:
        video_track_info_path = os.path.join(args.save_track1_dir, args.video_name+"-track1-info.pkl")
        if not os.path.exists(video_track_info_path):
            with open(video_track_info_path, "wb") as f:
                pickle.dump({"event":event_list, "frame_num":frame_num}, f)

    if args.mode == 'Track2':
        args.tube['triplet'][args.video_name] = event_list

    return 0


def make_t2_tube(tube, pre_cls):
    frames_len = len(tube['frames'])
    pre_cls = action_tube_padding(
        pre_cls,
        windeo_size=args.windows_size,
        frames_len=frames_len
    )
    for frame_num in range(frames_len):
        frame_key = "{:05d}.jpg".format(tube['frames'][frame_num])
        submit_pkl_json[args.video_name][frame_key][tube["track_id"]]["labels"].extend(list(pre_cls[frame_num]))
        
def track2(args):
    with torch.no_grad():
        with tqdm(args.tube['triplet'][args.video_name], desc="Processing tubes") as pbar:
            for t in pbar:
                classifier_dataset = Tracklet_Dataset(
                    mode='stage_two',
                    tracklet=stack_imgs_padding(t['stack_imgs'], args.windows_size), # TODO padding when frames_num < 4
                    args=args,
                    bbox=stack_boxes_padding(t['boxes'], args.windows_size)
                )

                pbar.set_description(f"Running T2 (number of tubes - {len(classifier_dataset)})")
                
                # predict
                pre_cls = []
                for stack_img, bbox in classifier_dataset:
                    input = torch.unsqueeze(stack_img, 0).to(int(args.devices))
                    bbox = torch.unsqueeze(bbox, 0).to(int(args.devices))
                    agent_ids = torch.zeros(bbox.size(0), bbox.size(1), 1)
                    agent_ids = (agent_ids + t['label_id']).long().to(int(args.devices))
                    pred = args.action_and_location_detector(input, bbox, agent_ids)
                    pred = torch.sigmoid(pred)
                    # cls = torch.argmax(pred, dim=1)
                    pred = list(pred.cpu().numpy().reshape(args.windows_size, -1))
                    pre_cls.extend(pred)

                # Padding and Matching t1 & t2 tubes
                make_t2_tube(t, pre_cls)
    return 0

def load_label_names(args):
    f = open(args.label_names_file, "r")
    labelnames = json.loads(f.read())
    
    agent_labels = labelnames["agent_names"]
    action_labels = labelnames["action_names"]
    loc_labels = labelnames["location_names"]
    f.close()
    return agent_labels, action_labels, loc_labels

def merge_two_tube(args, major_tube, rare_tube):
    """
    ToDo: Merge tube using IoU.

    """
    args.agent_labels, args.action_labels, args.loc_labels = load_label_names(args)
    rare_label_name = args.rare_yolo.names
    major_label_name = args.major_yolo.names
    for tube in rare_tube:
        tube['label_id'] = args.agent_labels.index(rare_label_name[tube['label_id']])
    for tube in major_tube:
        tube['label_id'] = args.agent_labels.index(major_label_name[tube['label_id']])
    
    # Merge submit_Json_Conf information in dict
    for tube_idx, tube in enumerate(rare_tube):
        for conf_idx, frame_id in enumerate(tube['frames']):
            frame_key = "{:05d}.jpg".format(frame_id)
            new_conf = [0.0 for i in range(len(args.agent_labels))]
            track_id = tube['track_id']
            item = submit_pkl_json[args.video_name][frame_key][track_id]
            for idx in sorted(args.rare_yolo.names, key=lambda k:int(k)):
                names = args.rare_yolo.names[idx]
                cls_id = args.agent_labels.index(names)
                new_conf[cls_id] = item['labels'][idx]
            submit_pkl_json[args.video_name][frame_key][track_id]['labels'] = new_conf
            rare_tube[tube_idx]['cls_conf'][conf_idx] = new_conf

    for tube_idx, tube in enumerate(major_tube):
        for conf_idx, frame_id in enumerate(tube['frames']):
            frame_key = "{:05d}.jpg".format(frame_id)
            new_conf = [0.0 for i in range(len(args.agent_labels))]
            track_id = tube['track_id']
            item = submit_pkl_json[args.video_name][frame_key][track_id]
            for idx in sorted(args.major_yolo.names, key=lambda k:int(k)):
                names = args.major_yolo.names[idx]
                cls_id = args.agent_labels.index(names)
                new_conf[cls_id] = item['labels'][idx]
            submit_pkl_json[args.video_name][frame_key][track_id]['labels'] = new_conf
            major_tube[tube_idx]['cls_conf'][conf_idx] = new_conf

    merged_tube = major_tube + rare_tube
    
    return merged_tube
    

def two_branch_yolo(args, video):
    args.tracker = args.major_yolo.track(
        source=video,
        imgsz=args.imgsz,
        device=args.devices,
        stream=True,
        conf = 0.0
    )
    major_tube, _ = make_tube(args)
    args.start_track_id = len(major_tube)

    args.tracker = args.rare_yolo.track(
        source=video,
        imgsz=args.imgsz,
        device=args.devices,
        stream=True,
        conf = 0.0
    )
    rare_tube, frame_num = make_tube(args)

    args.tube['triplet'][args.video_name] = merge_two_tube(args, major_tube, rare_tube)

    video_track_info_path = os.path.join(args.save_track1_dir, args.video_name+"-track1-info.pkl")
    if not os.path.exists(video_track_info_path):
        with open(video_track_info_path, "wb") as f:
            pickle.dump({"event":args.tube['triplet'][args.video_name], "frame_num":frame_num}, f)
    return 0


def main(args):
    args.tube = {
        'triplet': {}
    }
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    for v in sorted(glob.glob(os.path.join(args.video_path, '*.mp4'))):
        args.video_name = v.split('/')[-1].split('.')[0]
        if args.test_mode == "test" and args.video_name not in args.test_video_names:
            continue
        elif args.test_mode == "val_1" and args.video_name not in args.val_1_video_names:
            continue
        elif args.test_mode == "get_label" and args.video_name not in args.unlabel_train_1_video_names:
            continue
        elif args.test_mode == "get_label_train_1" and args.video_name not in args.labeled_train_1_video_names:
            continue
        
        # Check if this video has completed two stages of detection information, if so, skip
        output_video_path = os.path.join(args.outputdir, "{}.pkl".format(args.video_name))
        output_video_json_path = os.path.join(args.outputdir, "{}.json".format(args.video_name))
        if os.path.exists(output_video_path):
            with open(output_video_path, 'rb') as f:
                data = pickle.load(f)
                submit_pkl_json[args.video_name] = data
            continue

        # Check if this video has completed the first stage of detection information. If so, skip the first stage of detection
        if args.resume_track1:
            video_track_info_path = os.path.join(args.save_track1_dir, args.video_name+"-track1-info.pkl")
            if os.path.exists(video_track_info_path):
                with open(video_track_info_path, "rb") as f:
                    data = pickle.load(f)
                    frame_num = data['frame_num']
                    args.tube['triplet'][args.video_name] = data["event"]
                    if args.video_name not in submit_pkl_json.keys():
                        submit_pkl_json[args.video_name] = {}
                    for tube_idx, tube_data in enumerate(args.tube['triplet'][args.video_name]):
                        track_id = int(tube_data['track_id'])
                        for frame_idx, frame_id in enumerate(tube_data['frames']):
                            bbox = tube_data['boxes'][frame_idx]
                            labels = tube_data['cls_conf'][frame_idx]
                            frame_key = "{:05d}.jpg".format(frame_id)
                            if frame_key not in submit_pkl_json[args.video_name].keys():
                                submit_pkl_json[args.video_name][frame_key] = {}
                            submit_pkl_json[args.video_name][frame_key][track_id] = {
                                'bbox':bbox,
                                'labels':labels
                            }
                    for frame_id in range(frame_num):
                        frame_key = "{:05d}.jpg".format(frame_id + 1)
                        if frame_key not in submit_pkl_json[args.video_name].keys():
                            submit_pkl_json[args.video_name][frame_key] = []

            else:
                if args.video_name not in submit_pkl_json.keys():
                    submit_pkl_json[args.video_name] = {}
                if args.two_branch:
                    two_branch_yolo(args, v)
                else:
                    # tracking Using BoT-SORT
                    args.tracker = args.yolo.track(
                        source=v,
                        imgsz=args.imgsz,
                        device=args.devices,
                        stream=True,
                        conf = 0.0
                    )
                    make_tube(args)
                            

        # ToDo: two branch T2
        if args.mode == 'Track2':
            track2(args)
        
        # with open(output_video_json_path, 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(submit_pkl_json[args.video_name]))
        with open(output_video_json_path, 'wb') as f:
            pickle.dump(submit_pkl_json[args.video_name], f)

        for frame_key in submit_pkl_json[args.video_name].keys():
            new_obj_list = []
            if type(submit_pkl_json[args.video_name][frame_key]) is list:
                continue
            for track_id in submit_pkl_json[args.video_name][frame_key].keys():
                submit_pkl_json[args.video_name][frame_key][track_id]["labels"] = \
                    np.array(submit_pkl_json[args.video_name][frame_key][track_id]["labels"])
                new_obj_list.append(submit_pkl_json[args.video_name][frame_key][track_id])
            submit_pkl_json[args.video_name][frame_key] = new_obj_list
            
        # debug for one video
        with open(output_video_path, 'wb') as f:
            pickle.dump(submit_pkl_json[args.video_name], f)

    if args.save_res:
        outputpath = os.path.join(args.outputdir, args.pkl_name)
        if os.path.exists(outputpath):
            os.remove(outputpath)

        with open(outputpath, 'wb') as f:
            pickle.dump(submit_pkl_json, f)


if __name__ == '__main__':
    args = arg_parse()
    assert args.mode == 'Track1' or args.mode == 'Track2', 'detect mode only accept "Track1" or "Track2".'
    args.pkl_name = "yolov8_{}imgsz_{}class_{}winsz_vitclip-{}.pkl".format(args.imgsz[1], 11 if args.filter_agent else 10, args.windows_size, "two_branch" if args.two_branch else "one_branch") if args.pkl_name is None else args.pkl_name
    args.outputdir = os.path.join(args.outputdir, args.test_mode, args.pkl_name.split(".")[0], datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if args.resume is not None:
        args.outputdir = args.resume
    # debug_args:
    # args.devices = '0'
    device = torch.device("cuda:{}".format(args.devices) if torch.cuda.is_available() else "cpu")
    args.test_video_names = [
        "2014-06-26-09-31-18_stereo_centre_02",
        "2014-12-10-18-10-50_stereo_centre_02",
        "2015-02-03-08-45-10_stereo_centre_04",
        "2015-02-06-13-57-16_stereo_centre_01"
        ]
    args.val_1_video_names = [
        "2014-06-26-09-53-12_stereo_centre_02",
        "2014-11-25-09-18-32_stereo_centre_04",
        "2015-02-13-09-16-26_stereo_centre_02"
    ]
    args.unlabel_train_1_video_names = [
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
    args.labeled_train_1_video_names = [
        "2014-07-14-14-49-50_stereo_centre_01",
        "2015-02-03-19-43-11_stereo_centre_04",
        "2015-02-24-12-32-19_stereo_centre_04"
    ]

    if not os.path.exists(args.save_track1_dir):
        os.makedirs(args.save_track1_dir, exist_ok=True)
    args.save_track1_dir = os.path.join(args.save_track1_dir, args.test_mode, "two_branch" if args.two_branch else "one_branch", "yolov8-11class" if args.filter_agent else "yolov8-10class", args.yolo_name)
    if not os.path.exists(args.save_track1_dir):
        os.makedirs(args.save_track1_dir)
    args.start_track_id = 0 

    # two branch args:
    if args.two_branch:
        args.major_yolo = YOLO(args.major_path)
        args.rare_yolo = YOLO(args.rare_path)
    else:
        args.yolo = YOLO(args.yolo_path)
        print(args.yolo.names)
    
    if args.mode == 'Track2':
        args.action_and_location_detector = torch.load(args.classifier_path, map_location=torch.device("cuda:{}".format(args.devices)))
        args.action_and_location_detector.eval()
    
    submit_pkl_json = {}
    main(args)
