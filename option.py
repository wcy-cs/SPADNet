#-*- coding:UTF-8 -*-
import argparse
import os
import random
import torch
import numpy as np
parser = argparse.ArgumentParser(description='FaceSR')

parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--cuda_name', type=str, default='0')
parser.add_argument('--gpu_ids', type=int, default=1)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--device', default='cuda')
parser.add_argument('--dir_data', type=str, default='/data_c/zhwzhong/Data/wcy/data/CelebA',#'/data/disk_c/lllwcy/data/DIC/CelebA',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='train',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='test',
                    help='test dataset name')
parser.add_argument('--data_val', type=str, default='val',
                    help='val dataset name')
parser.add_argument('--scale', type=int, default=8,
                    help='super resolution scale')

parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--augment', action='store_true',
                    help='use data augmentation')
# Model specifications
parser.add_argument('--model', default='MYNET',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.2,
                    help='residual scaling')



parser.add_argument('--number', type=int, default=6,
                    help='concat attribute before the network')
parser.add_argument('--hab_num', type=int, default=5,
                    help='concat attribute before the network')


parser.add_argument('--epochs', type=int, default=2000,
                    help='number of epochs to train')

parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')

# Log specifications
# parser.add_argument('--root', type=str, default='/data/disk_c/lllwcy/code/FAGFSR')
parser.add_argument('--save', type=str, default='mynet',
                    help='file name to save')
parser.add_argument('--save_test', type=str, default='mynet',
                    help='file name to save test result')
parser.add_argument('--save_path', type=str, default='./experiment',
                    help='file path to save model')
parser.add_argument('--load_path', type=str, default='',
                    help='file name to load')

parser.add_argument("--writer_name", type=str, default="mynet",
                    help="the name of the writer")

args = parser.parse_args()

# args.scale = list(map(lambda x: int(x), args.scale.split('+')))
def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seed(args.seed)
if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

