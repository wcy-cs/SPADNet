import torch.nn as nn
import torch
import torch.nn.functional as F
from model.pac import PacConv2d


class HeatmapAttentionBlock35G2(nn.Module):
    def __init__(self, n_feats):
        super(HeatmapAttentionBlock35G2, self).__init__()
        self.conv1_SR = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1,
                              stride=1)  # conv(n_feats, n_feats, kernel_size, bias=bias))
        self.conv1_heatmap = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1, stride=1)
        self.conv2_heatmap = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1,
                                   stride=1)
        self.pacconv1 = PacConv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1,
                              dilation=1)
        self.conv2 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=n_feats*2, out_channels=n_feats, kernel_size=3, padding=1, stride=1)
        self.unfold = nn.Unfold(kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x, kernel):
        sr1 = self.conv1_SR(x)
        b, c, h, w = kernel.shape
        heatmap1 = self.conv1_heatmap(kernel)
        f1 = self.pacconv1(sr1, heatmap1)
        # print("type(f1): ", type(f1))
        heatmap1 = self.pool(heatmap1)
        heatmap2 = self.conv2_heatmap(heatmap1)
        global_bias = F.sigmoid(heatmap2)
        bias = global_bias.view(b, 64).unsqueeze(1)  # b,1,m
        bias = bias.repeat([1, h * w, 1])  # b,n_H*n_W,m
        bias = bias.reshape(b, c, h, w)
        bias = bias * sr1
        # res = F.conv2d(input=x, weight=bias, stride=(1, 1), padding=1)
        f = torch.cat((f1, bias), 1)
        f = self.conv3(f)
        f = self.relu(f)
        f1 = self.conv2(f)
        res = x + f1
        return res




class HeatmapAttentionBlock35G3(nn.Module):
    def __init__(self, n_feats):
        super(HeatmapAttentionBlock35G3, self).__init__()
        self.conv1_SR = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1,
                              stride=1)  # conv(n_feats, n_feats, kernel_size, bias=bias))
        self.conv1_heatmap = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1, stride=1)
        self.conv2_heatmap = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1,
                                   stride=1)
        self.pacconv1 = PacConv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1,
                              dilation=1)
        self.conv2 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=n_feats*2, out_channels=n_feats, kernel_size=3, padding=1, stride=1)
        self.unfold = nn.Unfold(kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)
        self.pool = nn.AdaptiveAvgPool2d(3)
    def forward(self, x, kernel):
        sr1 = self.conv1_SR(x)
        b, c, h, w = kernel.shape
        heatmap1 = self.conv1_heatmap(kernel)
        f1 = self.pacconv1(sr1, heatmap1)
        heatmap1 = self.pool(heatmap1)
        # print("heatmap1: ", heatmap1.shape)
        heatmap2 = self.conv2_heatmap(heatmap1)
        global_bias = F.sigmoid(heatmap2)
        # print("global_bias: ", global_bias.shape)
        bias = global_bias.view(b, 9, 64).squeeze(1)  # b,1,m
        # print("bias: ", bias.shape)
        bias = bias.repeat([1, h * w, 1])  # b,n_H*n_W,m
        # print("bia repeat: ", bias.shape)
        bias = bias.reshape(b, c,3*3, h, w)
        unfold_inputs = self.unfold(sr1).reshape(b, c, -1, h, w)
        # print("unfold_inputs: ", unfold_inputs.shape)
        f2 = (unfold_inputs * bias).sum(2)

        # res = F.conv2d(input=x, weight=bias, stride=(1, 1), padding=1)
        f = torch.cat((f1, f2), 1)
        f = self.conv3(f)
        f = self.relu(f)
        f1 = self.conv2(f)
        res = x + f1
        return res


