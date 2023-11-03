import pickle
import os
import numpy as np
import json
from tqdm import tqdm
from submit_requirement_for_task1 import make_require
def is_part_of_subsets(split_ids, SUBSETS):
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True
    return is_it

def update_agent_clsid(agent_id):
    agent_name = annojson['all_agent_labels'][agent_id]
    if agent_name in annojson['agent_labels']:
        agent_id = annojson['agent_labels'].index(agent_name)
    else:
        return -1
    return agent_id

def update_action_clsid(action_id):
    action_name = annojson['all_action_labels'][action_id]
    if action_name in annojson['action_labels']:
        action_id = annojson['action_labels'].index(action_name)
    else:
        return -1
    return action_id + 10

def update_location_clsid(location_id):
    return location_id + 29


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap*100

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

def get_gt_of_cls(gt_boxes, cls):
    cls_gt_boxes = []
    for i in range(gt_boxes.shape[0]):
        if len(gt_boxes.shape) > 1 and int(gt_boxes[i, -1]) == cls:
            cls_gt_boxes.append(gt_boxes[i, :-1])
    return np.asarray(cls_gt_boxes)

def evaluate_detections(gt_boxes, det_boxes, classes=[], iou_thresh=0.5):
    ap_strs = []
    num_frames = len(gt_boxes)
    print('Evaluating for '+ str(num_frames) + ' frames')
    ap_all = np.zeros(len(classes), dtype=np.float32)
    # loop over each class 'cls'
    for cls_ind, class_name in enumerate(tqdm(classes)):
        scores = np.zeros(num_frames * 2000)
        istp = np.zeros(num_frames * 2000)
        det_count = 0
        num_postives = 0.0
        for nf in range(num_frames):  # loop over each frame 'nf'
                # if len(gt_boxes[nf])>0 and len(det_boxes[cls_ind][nf]):
            # get frame detections for class cls in nf
            if nf in det_boxes[cls_ind].keys():
                frame_det_boxes = np.copy(det_boxes[cls_ind][nf])
            else:
                frame_det_boxes = np.array([])
            # get gt boxes for class cls in nf frame
            cls_gt_boxes = get_gt_of_cls(np.copy(gt_boxes[nf]), cls_ind)
            num_postives += cls_gt_boxes.shape[0]
            # check if there are dection for class cls in nf frame
            if frame_det_boxes.shape[0] > 0:
                # sort in descending order
                sorted_ids = np.argsort(-frame_det_boxes[:, -1])
                for k in sorted_ids:  # start from best scoring detection of cls to end
                    box = frame_det_boxes[k, :-1]  # detection bounfing box
                    score = frame_det_boxes[k, -1]  # detection score
                    ispositive = False  # set ispostive to false every time
                    # we can only find a postive detection
                    if cls_gt_boxes.shape[0] > 0:
                        # if there is atleast one gt bounding for class cls is there in frame nf
                        # compute IOU between remaining gt boxes
                        iou = compute_iou(box, cls_gt_boxes)
                        # and detection boxes
                        # get the max IOU window gt index
                        maxid = np.argmax(iou)
                        # check is max IOU is greater than detection threshold
                        if iou[maxid] >= iou_thresh:
                            ispositive = True  # if yes then this is ture positive detection
                            # remove assigned gt box
                            cls_gt_boxes = np.delete(cls_gt_boxes, maxid, 0)
                    # fill score array with score of current detection
                    scores[det_count] = score
                    if ispositive:
                        # set current detection index (det_count)
                        istp[det_count] = 1
                        #  to 1 if it is true postive example
                    det_count += 1
        if num_postives < 1:
            num_postives = 1
        scores = scores[:det_count]
        istp = istp[:det_count]
        argsort_scores = np.argsort(-scores)  # sort in descending order
        istp = istp[argsort_scores]  # reorder istp's on score sorting
        fp = np.cumsum(istp == 0)  # get false positives
        tp = np.cumsum(istp == 1)  # get  true positives
        fp = fp.astype(np.float64)
        tp = tp.astype(np.float64)
        recall = tp / float(num_postives)  # compute recall
        # compute precision
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # compute average precision using voc2007 metric
        cls_ap = voc_ap(recall, precision)
        ap_all[cls_ind] = cls_ap
        ap_str = class_name + ' : ' + \
            str(num_postives) + ' : ' + str(det_count) + ' : ' + str(cls_ap)
        ap_strs.append(ap_str)

    mAP = np.mean(ap_all)
    print('mean ap '+ str(mAP))
    print("agent mAP: ", ap_all[:10].mean())
    print("action mAP: ", ap_all[10:29].mean())
    print("location mAP: ", ap_all[29:41].mean())
    return mAP, ap_all, ap_strs


def load_label_names(label_names_file):
    f = open(label_names_file, "r")
    labelnames = json.loads(f.read())
    
    agent_labels = labelnames["agent_names"]
    action_labels = labelnames["action_names"]
    loc_labels = labelnames["location_names"]
    f.close()
    return agent_labels, action_labels, loc_labels

def get_gt_dict():
    gt_file = '/data/llk/projects/road-r/road-dataset/road/road_trainval_v1.0.json'
    f = open(gt_file, "r")
    annojson = json.loads(f.read())
    f.close()
    return annojson

def get_det_info():
    num_classes = 41
    all_det_info = [{} for i in range(num_classes)]
    frame_id = 0
    action_and_location_conf_thresh = 0.00
    for video_name in tqdm(val_1_video_names):
        for frame_key in sorted(data[video_name].keys(), key=lambda k:int(k.split(".")[0])):
            curframe_obj = data[video_name][frame_key]
            for tube_idx in range(len(curframe_obj)):
                x1, y1, x2, y2 = curframe_obj[tube_idx]["bbox"]
                labels_conf = curframe_obj[tube_idx]["labels"]
                # if max(labels_conf) < 0.5:
                #     continue
                for cls_id in range(num_classes):
                    if frame_id not in all_det_info[cls_id].keys():
                        all_det_info[cls_id][frame_id] = []
                    all_det_info[cls_id][frame_id].append([x1, y1, x2, y2,  0 if cls_id >= 10 and labels_conf[cls_id] < action_and_location_conf_thresh else labels_conf[cls_id]])
            for cls_id in range(num_classes):
                if frame_id in all_det_info[cls_id].keys():
                    all_det_info[cls_id][frame_id] = np.array(all_det_info[cls_id][frame_id])
            frame_id += 1
    return all_det_info

def get_gt_info():
    frame_id = 0
    all_gt_info = {}
    w, h = 1280, 960
    for video_name in tqdm(val_1_video_names):
        for frame_key in sorted(annojson["db"][video_name]["frames"].keys(), key=lambda k:int(k)):
            curframe_obj = annojson["db"][video_name]["frames"][frame_key]
            if frame_id not in all_gt_info.keys():
                all_gt_info[frame_id] = []
            if "annos" in curframe_obj.keys():
                if len(curframe_obj["annos"]) > 0:
                    for tube_k in curframe_obj["annos"].keys():
                        tube_item = curframe_obj["annos"][tube_k]
                        x1, y1, x2, y2 = tube_item['box']
                        x1, y1, x2, y2 = x1*w, y1*h, x2*w, y2*h
                        for agent_id in tube_item["agent_ids"]:
                            new_agent_id = update_agent_clsid(agent_id)
                            if new_agent_id == -1:
                                continue
                            all_gt_info[frame_id].append([x1, y1, x2, y2, new_agent_id])
                        for action_id in tube_item["action_ids"]:
                            new_action_id = update_action_clsid(action_id)
                            if new_action_id == -1:
                                continue
                            all_gt_info[frame_id].append([x1, y1, x2, y2, new_action_id])
                        for location_id in tube_item["loc_ids"]:
                            new_location_id = update_location_clsid(location_id)
                            if new_location_id == -1:
                                continue
                            all_gt_info[frame_id].append([x1, y1, x2, y2, new_location_id])
                else:
                    all_gt_info[frame_id] = []
            else:
                all_gt_info[frame_id] = []
            all_gt_info[frame_id] = np.array(all_gt_info[frame_id])
            frame_id += 1
    return all_gt_info

if __name__ == '__main__':
    task1_labels = [
        "2014-07-14-14-49-50_stereo_centre_01",
        "2015-02-03-19-43-11_stereo_centre_04",
        "2015-02-24-12-32-19_stereo_centre_04"
        ]
    label_names_file = "configs/label_names.json"
    annojson = get_gt_dict()
    all_video_names = annojson["db"].keys()
    val_1_video_names = []
    for vname in all_video_names:
        if is_part_of_subsets(annojson["db"][vname]["split_ids"], ["val_1"]) and vname not in task1_labels:
            val_1_video_names.append(vname)
    all_gt_info = get_gt_info()
    agent_labels, action_labels, loc_labels = load_label_names(label_names_file)
    class_names = []
    class_names.extend(agent_labels)
    class_names.extend(action_labels)
    class_names.extend(loc_labels)

    # 读取.pkl文件
    print("before logical requirements...")
    before_require_pkl = r'output/val_1/val_1_vitcliplarge-base-yolov8-bestacc/2023-11-02-15-38-30/val_1_vitcliplarge-base-yolov8-bestacc.pkl'
    # before_require_pkl = "submit-task1.pkl"
    with open(before_require_pkl, 'rb') as f:
        data = pickle.load(f)
    all_before_require_det_info = get_det_info()
    
    mAP, ap_all, ap_strs = evaluate_detections(gt_boxes=all_gt_info, det_boxes=all_before_require_det_info, classes=class_names, iou_thresh=0.5)

    after_require_pkl = 'utils/submit-require-task1.pkl'
    # after_require_pkl = 'utils/submit-baseline-require-task1.pkl'
    
    print("make logical requirements for task1...")
    make_require(before_pkl=before_require_pkl, after_pkl=after_require_pkl)
    print("==="*10)
    print("after logical requirements...")
    with open(after_require_pkl, 'rb') as f:
        data = pickle.load(f)
    all_after_require_det_info = get_det_info()
    mAP, ap_all, ap_strs = evaluate_detections(gt_boxes=all_gt_info, det_boxes=all_after_require_det_info, classes=class_names, iou_thresh=0.5)



    # f = open("./result.txt", "w", encoding="utf-8")
    # for line in ap_strs:
    #     print(line)
    #     line += '\n'
    #     f.write(line)
    # f.close()