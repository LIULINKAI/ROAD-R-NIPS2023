import argparse

def arg_parse(func):
    parser = argparse.ArgumentParser()

    if func == "main":    
        parser.add_argument('--gpu_num', nargs='+', type=str, default="0", help='target gpu number')
        parser.add_argument('--deviceid', type=str, default="7", help='cuda device id')
        parser.add_argument('--parallelism', type=bool, default=False, help='Implements data parallelism')
        parser.add_argument('--dataset_path', '-dpath', default="base-datasets/act-and-loc-datasets/datasets", help='path of dataset')
        parser.add_argument('--global_img_path', default="~/projects/road-r/road-dataset/road/rgb-images")
        parser.add_argument('--label_names_file', default="configs/label_names.json", help='path of label names')
        parser.add_argument('--num_workers', '-nwork', type=int, default=8, help='path of dataset')
        parser.add_argument('--resume', '-re', type=str, default=0, help='path of dataset')
        parser.add_argument('--expname', type=str, default=None, help='path of dataset')
        
        parser.add_argument('--one_output', type=bool, default=False)
        parser.add_argument('--return_agent', type=bool, default=True)
        parser.add_argument('--target', default="stage2", help='action or location')
        parser.add_argument('--window_size', '-wsize',  default=4, help='num of frames')
        parser.add_argument('--input_shape', '-inshape', nargs='+', default=(224, 224), help='path of dataset')

        parser.add_argument('--model', default='vit_clip_pro', help='model name')
        parser.add_argument('--head_mode', default='2d', help='model name')
        parser.add_argument('--pretrain', default='', help='pretrain weight path')
        parser.add_argument('--epoch', type=int, default=10, help='number of epoch to train')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning_rate')
        parser.add_argument("--batch_size", type=int, default=2, help='the batch for id')
        parser.add_argument("--seed", type=int, default=0, help='random seed')

        parser.add_argument("--use_local", action='store_true', default=False, help='debug switch')

        parser.add_argument("--debug", type=bool, default=False, help='debug switch')
        parser.add_argument('--alpha', type=float, default=0.25, help='focal loss')
        parser.add_argument('--gamma', type=float, default=2.0, help='focal loss')
        parser.add_argument('--acc_pos_thresh', type=float, default=0.25, help='focal loss')
    opt = parser.parse_args()
    return opt