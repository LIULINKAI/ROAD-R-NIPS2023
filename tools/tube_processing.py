import pickle
import numpy as np

    
def bbox_normalized(bbox, img_w, img_h):
    return bbox / np.array([img_w, img_h, img_w, img_h])


def norm_box_into_absolute(bbox, img_w, img_h):
    return bbox * np.array([img_w, img_h, img_w, img_h])


def tube_change_axis(tube, orig_shape, submit_shape):
    ori_h, ori_w = orig_shape
    new_h, new_w = submit_shape
    
    tube['boxes'] = np.array([norm_box_into_absolute(bbox_normalized(box, ori_w, ori_h), new_w, new_h) for box in tube['boxes']])
    
    return tube


def pkl_change_axis(tubes, ori_w, ori_h, new_w, new_h):
    for video, tube in tubes.items():
        for t in tube:
            t['boxes'] = np.array([norm_box_into_absolute(bbox_normalized(box, ori_w, ori_h), new_w, new_h) for box in t['boxes']])
            
    return tubes


def action_tube_padding(action_cls, windeo_size, frames_len):
    if frames_len < windeo_size:
        maxlen = windeo_size-frames_len
        prelen = maxlen // 2
        return action_cls[prelen:prelen+frames_len]
    else:
        step_len = len(action_cls) // windeo_size
        result = [np.zeros_like(action_cls[0]) for i in range(frames_len)]
        num_idx = [0 for i in range(frames_len)]
        for i in range(frames_len):
            for j in range(windeo_size):
                if j+windeo_size*i >= len(action_cls):
                    continue
                result[i+j] += action_cls[j+windeo_size*i]
                num_idx[i+j] += 1
            result[i] /= num_idx[i]
        return result


def stack_imgs_padding(stack_imgs, windeo_size=8):
    if len(stack_imgs) < windeo_size:
        maxlen = windeo_size-len(stack_imgs)
        prelen = maxlen // 2
        if prelen > 0:
            stack_imgs = ([stack_imgs[0]] * prelen) + stack_imgs
        if maxlen - prelen > 0:
            stack_imgs = stack_imgs + ([stack_imgs[-1]] * (maxlen - prelen))
        return stack_imgs
    else:
        return stack_imgs    

def stack_boxes_padding(boxes, windeo_size=8):
    boxes = list(boxes)
    if len(boxes) < windeo_size:
        maxlen = windeo_size-len(boxes)
        prelen = maxlen // 2
        if prelen > 0:
            boxes = ([boxes[0]] * prelen) + boxes
        if maxlen - prelen > 0:
            boxes = boxes + ([boxes[-1]] * (maxlen - prelen))
        return boxes
    else:
        return boxes    
