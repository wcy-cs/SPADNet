import torch.nn as nn
import torch
from model.common import *
from model.hab import *
import torch.nn.functional as F



def hab_block(args, n_feats=64):
    mm = []

    for i in range(args.hab_num):
        mm.append(HeatmapAttentionBlock35G2(n_feats))

    return nn.Sequential(*mm)


def basic_block(args,
                 ):
    return nn.Sequential(*[RCAB(n_feats=args.n_feats) for i in range(args.number)])


class DownBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.scale, self.scale, w // self.scale, self.scale)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.scale ** 2), h // self.scale, w // self.scale)
        return x

class HAPFSR(nn.Module):
    def __init__(self, args, n_feats=64, number=6, embed_dim=64, num_head=2, mlp_fusion='mlp'):
        super(HAPFSR, self).__init__()
        self.args = args
        self.mlp_fusion = mlp_fusion

        modules_head = [
            DownBlock(args.scale),
            nn.Conv2d(3 * args.scale ** 2, args.n_feats, 3, 1, 1),
        ]
        self.head = nn.Sequential(*modules_head)

        # # # SR branch # # #

        self.SR_branch1 = basic_block(args=args)
        self.SR_up1 = nn.Sequential(*[Upsampler(conv=default_conv, scale=2, n_feats=n_feats)])


        self.SR_branch2 = basic_block(args=args)
        self.SR_up2 = nn.Sequential(*[Upsampler(conv=default_conv, scale=2, n_feats=n_feats)])


        self.SR_branch3 = basic_block(args=args)
        self.SR_up3 = nn.Sequential(*[Upsampler(conv=default_conv, scale=2, n_feats=n_feats)])
        self.SR_out = nn.Conv2d(in_channels=n_feats, out_channels=3, kernel_size=3, padding=1, stride=1)

        self.hab_branch1 = hab_block(args)
        self.hab_branch2 = hab_block(args)
        self.hab_branch3 = hab_block(args)

        # # # Heatmap branch # # #

        self.Heatmap_branch1 = basic_block(args=args)
        self.Heatmap_up1 = nn.Sequential(*[Upsampler(conv=default_conv, scale=2, n_feats=n_feats)])

        self.Heatmap_branch2 = basic_block(args=args)
        self.Heatmap_up2 = nn.Sequential(*[Upsampler(conv=default_conv, scale=2, n_feats=n_feats)])

        self.Heatmap_branch3 = basic_block(args=args)
        self.Heatmap_up3 = nn.Sequential(*[Upsampler(conv=default_conv, scale=2, n_feats=n_feats)])
        self.Heatmap_out = nn.Conv2d(in_channels=n_feats, out_channels=5, kernel_size=3, padding=1, stride=1)


    def forward(self, x):
        feature = self.head(x)
        # # # SR branch # # #
        sr_feature = feature
        heatmap_feature = feature


        # # # SR step 1 # # #

        for i in range(len(self.SR_branch1) * 2 - 1):
            if i % 2 == 0:
                sr_feature = self.SR_branch1[i // 2](sr_feature)
                heatmap_feature = self.Heatmap_branch1[i // 2](heatmap_feature)
            else:

                sr_feature = self.hab_branch1[i // 2](sr_feature, heatmap_feature)


        sr_feature = self.SR_up1(sr_feature) + F.interpolate(feature, scale_factor=2, mode='bicubic')

        # # # Heatmap step 1 # # #
        heatmap_feature = self.Heatmap_up1(heatmap_feature)

        # # # SR step 2 # # #
        for i in range(len(self.SR_branch2) * 2 - 1):
            if i % 2 == 0:
                sr_feature = self.SR_branch2[i // 2](sr_feature)
                heatmap_feature = self.Heatmap_branch2[i // 2](heatmap_feature)
            else:

                sr_feature = self.hab_branch2[i // 2](sr_feature, heatmap_feature)


        sr_feature = self.SR_up2(sr_feature)+ F.interpolate(feature, scale_factor=4, mode='bicubic')

        heatmap_feature = self.Heatmap_up2(heatmap_feature)
        # # # SR step 3 # # #
        for i in range(len(self.SR_branch2) * 2 - 1):
            if i % 2 == 0:
                sr_feature = self.SR_branch3[i // 2](sr_feature)
                heatmap_feature = self.Heatmap_branch3[i // 2](heatmap_feature)
            else:
                sr_feature = self.hab_branch3[i // 2](sr_feature, heatmap_feature)

        sr_feature = self.SR_up3(sr_feature)+ F.interpolate(feature, scale_factor=8, mode='bicubic')
        heatmap_feature = self.Heatmap_up3(heatmap_feature)
        sr = self.SR_out(sr_feature)
        heatmap = self.Heatmap_out(heatmap_feature)

        return sr, heatmap

def make_model(args):
    return HAPFSR(args)



