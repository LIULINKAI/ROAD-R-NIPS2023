import pickle
import numpy as np
from tqdm import tqdm


def tube_interpolation(tube):
    frames = tube['frames']
    scores = tube['scores']
    boxes = tube['boxes']
    
    interpolated_frames = np.arange(frames[0], frames[-1] + 1)  
    interpolated_scores = np.interp(interpolated_frames, frames, scores)  
    interpolated_boxes = np.empty((len(interpolated_frames), 4))  
    
    for i, axis in enumerate([0, 1, 2, 3]):
        interpolated_boxes[:, i] = np.interp(interpolated_frames, frames, boxes[:, axis])
    
    interpolated_tube = {
        'label_id': tube['label_id'],
        'scores': interpolated_scores,
        'boxes': interpolated_boxes,
        'score': tube['score'],
        'frames': interpolated_frames
    }
    
    return interpolated_tube
