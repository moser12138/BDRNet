import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import logging
import argparse
import math
from tabulate import tabulate

from tqdm import tqdm
import numpy as np
import cv2

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from lib.model import model_factory

from configs import set_cfg_from_file
from lib.logger import setup_logger
from lib.data import get_data_loader
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# 数据预处理和转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 数据加载器
batch_size = 1  # 批大小为1，因为我们要逐个样本计算像素准确率


# 计算像素准确率
def calculate_pixel_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total
    return accuracy

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='model_coco_29_86.pth',)
    parse.add_argument('--config', dest='config', type=str, default='configs/camvid.py',)
    return parse.parse_args()

def main():
    args = parse_args()
    cfg = set_cfg_from_file(args.config)
    # cfg = set_cfg_from_file('configs/camvid.py')
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-eval', cfg.respth)

    net = model_factory[cfg.model_type](cfg.n_cats)
    net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'))
    net.cuda()

    # 模型评估
    net.eval()
    total_pixel_accuracy = 0.0
    num_samples = 0
    dataset = get_data_loader(cfg, mode='val')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for images, labels in data_loader:
        with torch.no_grad():
            predictions = net(images)
            pixel_accuracy = calculate_pixel_accuracy(predictions, labels)
            total_pixel_accuracy += pixel_accuracy
            num_samples += 1

    # 计算平均像素准确率
    average_pixel_accuracy = total_pixel_accuracy / num_samples
    print("Average Pixel Accuracy:", average_pixel_accuracy)


if __name__ == "__main__":
    main()