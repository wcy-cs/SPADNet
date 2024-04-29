from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy
import torch
import glob
import random
import numpy as np
import pickle
from data import util
def get_landmark(landmark_dict_path, paths_HR):

    landmark_list = []
    with open(landmark_dict_path, 'rb') as f:
        landmark_dict = pickle.load(f)

    for p in paths_HR:
        img_name = os.path.basename(p)
        # print(img_name)
        landmark_list.append(landmark_dict[img_name])
    return landmark_list

def augment(lr, hr, parsing, hflip=True, rot=True):


    if random.random() > 0.5 and hflip:
        lr = lr[:, ::-1, :]
        hr = hr[:, ::-1, :]
        # print("hflip")
        org = parsing
        parsing = parsing[:, :, ::-1]
        #landmark[:, 0] = hr.shape[1] - landmark[:, 0]
        # print("hflip: ", org.shape, parsing.shape)

    if rot:
        rot_rand = random.random()
        if rot_rand > 0.75:
            lr = np.rot90(lr, k=1, axes=(0, 1))
            hr = np.rot90(hr, k=1, axes=(0, 1))
            org = parsing
            parsing = np.rot90(parsing, k=1, axes=(0, 1))

        elif rot_rand > 0.5:
            lr = np.rot90(lr, k=2, axes=(0, 1))
            hr = np.rot90(hr, k=2, axes=(0, 1))
            org = parsing
            parsing = np.rot90(parsing, k=2, axes=(0, 1))

        elif rot_rand > 0.25:
            lr = np.rot90(lr, k=3, axes=(0, 1))
            hr = np.rot90(hr, k=3, axes=(0, 1))
            org = parsing
            parsing = np.rot90(parsing, k=3, axes=(0, 1))

    return lr, hr, parsing


class Data(data.Dataset):
    def __init__(self, root, args, train=False):
        # 返回指定路径下的文件和文件夹列表。
        self.args = args
        self.root = root
        self.imgs_HR_path = os.path.join(root, 'HR')
        self.imgs_HR = sorted(
            glob.glob(os.path.join(self.imgs_HR_path, '*.png'))
        )

        if self.args.scale == 8:
            self.imgs_LR_path = os.path.join(root, 'LR_bicubic')
        elif self.args.scale == 4:
            self.imgs_LR_path = os.path.join(root, 'LR_x4_bicubic')
        elif self.args.scale == 16:
            self.imgs_LR_path = os.path.join(root, 'LR_x16_bicubic')


        self.imgs_LR = sorted(
            glob.glob(os.path.join(self.imgs_LR_path, '*.png'))
        )

        self.landmarks = get_landmark(os.path.join(root, 'landmark.pkl'), self.imgs_HR)

        self.transform = transforms.ToTensor()
        self.train = train

    def __getitem__(self, item):

        img_path_LR = os.path.join(self.imgs_LR_path, self.imgs_LR[item])
        img_path_HR = os.path.join(self.imgs_HR_path, self.imgs_HR[item])
        landmark = self.landmarks[item]
        landmark = numpy.array(landmark)

        LR = Image.open(img_path_LR)
        HR = Image.open(img_path_HR)

        HR = numpy.array(HR)
        LR = numpy.array(LR)

        # print("1")
        filename = os.path.basename(img_path_HR)


        if "test" not in self.root :
            landmark = landmark[0]
        else:
            landmark = landmark


        heatmap = util.generate_gt(128, landmark)


        new_heatmap = np.zeros_like(heatmap[:5, :, :])
        new_heatmap[0, :, :] = heatmap[36:42, :, :].sum(0)  # left eye
        new_heatmap[1, :, :] = heatmap[42:48, :, :].sum(0)  # right eye
        new_heatmap[2, :, :] = heatmap[27:36, :, :].sum(0)  # nose
        new_heatmap[3, :, :] = heatmap[48:68, :, :].sum(0)  # mouse
        new_heatmap[4, :, :] = heatmap[:27, :, :].sum(0)  # face silhouette

        new_heatmap = numpy.array(new_heatmap)
        new_heatmap = np.transpose(new_heatmap, (2, 1, 0))
        if self.args.augment and self.train:

            LR, HR, new_heatmap = augment(LR, HR, new_heatmap)
        LR = np.ascontiguousarray(LR)
        HR = np.ascontiguousarray(HR)
        new_heatmap = np.transpose(new_heatmap, (2, 0, 1))
        gt_heatmap = torch.from_numpy(np.ascontiguousarray(new_heatmap))

        HR = ToTensor()(HR)
        LR = ToTensor()(LR)

        gt_heatmap = gt_heatmap.float()

        return LR, HR, gt_heatmap, filename


    def __len__(self):
        return len(self.imgs_HR)
