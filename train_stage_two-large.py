import os
import copy
import json
import time
import shutil
import random
import logging
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix
import datetime
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from models.vitCLIP_base import *
from models.vitCLIP_pro import *
from tools.dataset import mydataset
from tools.loss import sigmoid_focal_loss
from tools.engine_stage_two import train as train_loc
from tools.engine_stage_two import test as test_loc
import argparse

def arg_parse(func):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', nargs='+', type=str, default="0", help='target gpu number')
    parser.add_argument('--deviceid', type=str, default="3", help='cuda device id')
    parser.add_argument('--parallelism', type=bool, default=False, help='Implements data parallelism')
    parser.add_argument('--dataset_path', '-dpath', default="base-datasets/act-and-loc-datasets/datasets", help='path of dataset')
    parser.add_argument('--global_img_path', default="../../../road-dataset/road/rgb-images")
    parser.add_argument('--label_names_file', default="configs/label_names.json", help='path of label names')
    parser.add_argument('--num_workers', '-nwork', type=int, default=8, help='path of dataset')
    parser.add_argument('--resume', '-re', type=str, default=0, help='path of dataset')
    parser.add_argument('--expname', type=str, default=None, help='path of dataset')
    
    parser.add_argument('--one_output', type=bool, default=False)
    parser.add_argument('--return_agent', type=bool, default=False)
    parser.add_argument('--target', default="stage2", help='action or location')
    parser.add_argument('--window_size', '-wsize',  default=4, help='num of frames')
    parser.add_argument('--input_shape', '-inshape', nargs='+', default=(224, 224), help='path of dataset')

    parser.add_argument('--model', default='vit_clip_pro', help='model name')
    parser.add_argument('--head_mode', default='2d', help='model name')
    parser.add_argument('--pretrain', default='', help='pretrain weight path')
    parser.add_argument('--epoch', type=int, default=3, help='number of epoch to train')
    parser.add_argument('--start_epoch', type=int, default=1, help='number of epoch to train')
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

def load_label_names(args):
    f = open(args.label_names_file, "r")
    labelnames = json.loads(f.read())
    
    agent_labels = labelnames["agent_names"]
    action_labels = labelnames["action_names"]
    loc_labels = labelnames["location_names"]
    f.close()
    return agent_labels, action_labels, loc_labels


def torch_init(args):
    # including random_split
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


def logger_init(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    handler1 = logging.FileHandler("./runs/{}/train_log.log".format(train_id))
    handler2 = logging.StreamHandler()
    formatter = logging.Formatter(str(train_id) + ': %(asctime)s - %(levelname)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    logger.info("{} process start.".format(train_id))
    logger.info("--- option argument setting ---")
    logger.info("gpu_num = {}".format(args.gpu_num))
    logger.info("parallelism = {}".format(args.parallelism))
    logger.info("dataset_path = {}".format(args.dataset_path))
    logger.info("epoch = {}".format(args.epoch))
    logger.info("lr = {}".format(args.lr))
    logger.info("batch_size = {}".format(args.batch_size))
    logger.info("seed = {}".format(args.seed))
    logger.info("-------------------------------")

    return logger


def model_init(args):
    if args.resume:
        model = torch.load("./runs/{}/weight/best_acc_weight.pt".format(args.resume))
    else:
        args.input_shape = [int(args.input_shape[0]), int(args.input_shape[1])]
        if "swin" in args.model or "vit" in args.model:
            assert args.input_shape[0] == args.input_shape[1]

        if args.model == 'vit_clip': # broken
            model = ViTCLIPClassifier_location_base(
                input_resolution=args.input_shape[0],
                num_frames=args.window_size, 
                patch_size=16,
                width=768,
                layers=12,
                heads=12,
                num_classes=args.num_class,
                drop_path_rate=0.2, 
                adapter_scale=0.5,
                head_type=args.head_mode,
                pretrained="ViT-B/16",
                use_local=args.use_local
                )
        elif args.model == 'vit_clip_pro':
            model = ViTCLIPClassifier_location_pro(
                input_resolution=args.input_shape[0],
                num_frames=int(args.window_size), 
                patch_size=14,
                width=1024,
                layers=24,
                heads=16,
                num_classes=args.num_class,
                drop_path_rate=0.2, 
                adapter_scale=0.5,
                head_type=args.head_mode,
                pretrained="ViT-L/14",
                use_local=args.use_local
                )
    for name, param in model.named_parameters():
        if 'to_hide' not in name and 'location_embedding' not in name and 'agent_embedding' not in name and 'bbox_embedding' not in name and 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name:
            param.requires_grad = False
    for name, param in model.named_parameters():
        print('{}: {}'.format(name, param.requires_grad))
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    return model


def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)

    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)

    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)), normalize=normalize
    )

    return cmtx


def plot_confusion_matrix(cmtx, num_classes, cls_names=None, figsize=None):
    if cls_names is None or type(cls_names) != list:
        cls_names = [str(i) for i in range(num_classes)]

    fig = plt.figure(figsize=figsize)

    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(cls_names))
    plt.xticks(tick_marks, cls_names, rotation=45)
    plt.yticks(tick_marks, cls_names)

    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j, i, format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".", 
            horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    return fig


def add_confusion_matrix(writer, cmtx, num_classes, global_step=None, subset_ids=None,
                         class_names=None, tag="Confusion Matrix", figsize=None):
    if subset_ids is None or len(subset_ids) != 0:
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = plot_confusion_matrix(
            sub_cmtx,
            num_classes=len(subset_ids),
            cls_names=sub_names,
            figsize=figsize
        )

        writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)


def main():
    output_path = './runs/{}/'.format(train_id)

    # check folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'weight'))

    # init
    logger = logger_init(args)
    # if not args.cpu:
    #     gpu_list = GPU_init(args, logger)
    torch_init(args)
    writer = SummaryWriter(log_dir=output_path)

    # model
    model = model_init(args)
    if args.parallelism:
        model = torch.nn.DataParallel(model, device_ids=device_id)
    model = model.to(args.device)

    logger.info("loading train data...")
    logger.info("train test split ratio: 0.7")
    logger.info("mini batch size: 0.1")

    train_set = mydataset(args, is_train=True, use_local=args.use_local, return_agent=args.return_agent)
    valid_set = mydataset(args, is_train=False, use_local=args.use_local, return_agent=args.return_agent)

    train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size = args.batch_size, 
            shuffle = True,
            num_workers = args.num_workers,
            pin_memory = False
        )
    test_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = False
    )
    
    logger.info("optimizer: Adam")
    logger.info("Loss function: CrossEntropyLoss")
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = sigmoid_focal_loss

    logger.info("----------------")
    best_loss, best_acc = 999., 0.0
    for epoch in range(args.start_epoch, args.epoch+1):
        train_loss, train_acc = train_loc(args, model, train_loader, optimizer, criterion, epoch, writer)
        # testing
        test_loss, test_acc, preds, labels = test_loc(args, model, test_loader, criterion, epoch)

        if best_loss > test_loss:
            best_loss = test_loss
            torch.save(model, "./runs/{}/weight/best_weight.pt".format(train_id))
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model, "./runs/{}/weight/best_acc_weight.pt".format(train_id))

        logger.info("epoch {}, test loss: {:.6f}, test acc: {:.6f} || best loss:{:.6f}, best acc:{:.6f}".format(epoch, test_loss, test_acc, best_loss, best_acc))

        # tensorboard loss and acc
        writer.add_scalars(main_tag="Loss History", tag_scalar_dict={
            "Train_Loss": train_loss,
            "Valid_Loss": test_loss
        }, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy History", tag_scalar_dict={
            "Train_Acc": train_acc,
            "Valid_Acc": test_acc
        }, global_step=epoch)

        # tensorboard confusion matrix
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        cmtx = get_confusion_matrix(preds, labels, len(args.action_labels if args.target == "action" else args.loc_labels))
        add_confusion_matrix(writer, cmtx, num_classes=len(args.action_labels if args.target == "action" else args.loc_labels), global_step=epoch,
                                class_names=args.action_labels if args.target == "action" else args.loc_labels, tag="Test Confusion Matrix", figsize=[10, 8])

        logger.disabled = True
        logger.info("Epoch {}, train_loss: {:.6f}, test_loss: {:.6f}".format(epoch, train_loss, test_loss))
        logger.disabled = False

    writer.close()
    total_time = (int(time.time()) - start_time) // 60
    h_time, m_time = (total_time // 60), total_time % 60
    logger.info("Total prossesing time = {}:{}:{}".format(h_time, m_time, int(time.time()) - total_time*60))
    logger.info("{} process end.".format(train_id))


if __name__ == '__main__':
    args = arg_parse("main")
    args.device = torch.device("cuda:{}".format(args.deviceid) if torch.cuda.is_available() else "cpu")
    start_time = int(time.time())
    # train_id = "debug" if args.debug else start_time % 100000

    train_id = "exp-{}-{}-".format(args.target, args.model) + datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    train_id = train_id if args.expname is None else args.expname + datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    train_id = args.resume if args.resume else train_id

    if len(args.gpu_num) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num[0]
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpu_string = args.gpu_num[0]
        for i in args.gpu_num[1:]:
            gpu_string = gpu_string + ", " + i
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_string
    device_id = [int(i) for i in range(len(args.gpu_num))]
    args.agent_labels, args.action_labels, args.loc_labels = load_label_names(args)
    args.num_class = len(args.action_labels) + len(args.loc_labels)
    main()
