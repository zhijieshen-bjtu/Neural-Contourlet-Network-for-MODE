
import torch
import torch.nn as nn
from utils.layers import *
from utils.contourlet_torch import ContourDec, ContourRec

from collections import OrderedDict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
#from skimage import metrics
import math

# class SGM(nn.Module):
#     def __init__(self, ):
#         super(SGM, self).__init__()
#
#         wave_pad = "zero"
#         padding = "reflection"
#         self.wt = DWT(wave="haar", mode=wave_pad)
#
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, gray):
#         self.hh = []
#         gray_ll0, h0 = self.wt(gray)
#         # img_ = torch.mean(h0[0], dim=0, keepdim=True)
#         # plt.imshow(np.transpose(img_.cpu().numpy(), (1, 2, 0)))
#         # plt.show()
#         self.hh.append(torch.mean(h0[0], dim=1))
#         gray_ll1, h1 = self.wt(gray_ll0)
#         self.hh.append(torch.mean(h1[0], dim=1))
#         gray_ll2, h2 = self.wt(gray_ll1)
#         self.hh.append(torch.mean(h2[0], dim=1))
#         gray_ll3, h3 = self.wt(gray_ll2)
#         self.hh.append(torch.mean(h3[0], dim=1))
#
#
#         return self.hh


def get_mask(hh):
    b, c, h, w = hh[0].shape
    mask = []
    hhf = torch.cat((hh[0], hh[1], hh[2], hh[3]), dim=1)
    absf = torch.abs(hhf)
    meanf, _ = torch.median(absf.flatten(2), dim=2)
    b, c = meanf.shape
    th = torch.full((b, c), 2 * math.log(h * w)).to(hh[0].device)#torch.from_numpy(2 * np.log(h * w, dtype=float)).unsqueeze(0)
    #th = th.repeat(b, c)
    meanf = torch.sqrt(th) * meanf / 0.6745
    meanf = meanf.unsqueeze(2).unsqueeze(3)
    meanf = meanf.repeat(1, 1, hh[0].shape[-2], hh[0].shape[-1])
    hhf[torch.abs(hhf) < meanf] = 0
    hhf[torch.abs(hhf) >= meanf] = 1
    hhb = torch.cat((hh[4], hh[5], hh[6], hh[7]), dim=1)
    absb = torch.abs(hhb)
    meanb, _ = torch.median(absb.flatten(2), dim=2)
    b, c = meanb.shape
    th = torch.full((b, c), 2 * math.log(h * w)).to(hh[0].device)  # torch.from_numpy(2 * np.log(h * w, dtype=float)).unsqueeze(0)
    #th = th.repeat(b, c)
    meanb = torch.sqrt(th) * meanb / 0.6745
    meanb = meanb.unsqueeze(2).unsqueeze(3)
    meanb = meanb.repeat(1, 1, hh[4].shape[-2], hh[4].shape[-1])
    hhb[torch.abs(hhb) < meanb] = 0
    hhb[torch.abs(hhb) >= meanb] = 1
    mask.append(hhf[:, 0, :, :].unsqueeze(1))
    mask.append(hhf[:, 1, :, :].unsqueeze(1))
    mask.append(hhf[:, 2, :, :].unsqueeze(1))
    mask.append(hhf[:, 3, :, :].unsqueeze(1))
    mask.append(hhb[:, 0, :, :].unsqueeze(1))
    mask.append(hhb[:, 1, :, :].unsqueeze(1))
    mask.append(hhb[:, 2, :, :].unsqueeze(1))
    mask.append(hhb[:, 3, :, :].unsqueeze(1))
    # hhb = torch.cat((hh[4], hh[5], hh[6], hh[7]), dim=1)
    # absb = torch.abs(hhb)
    # meanb = torch.mean(absb, dim=1)

    return mask


def get_mask2(hh):
    b, c, h, w = hh[0].shape
    mask = []
    hhf = torch.cat((hh[0], hh[1], hh[2], hh[3]), dim=1)
    absf = torch.abs(hhf)
    meanf, _ = torch.median(absf.flatten(2), dim=2)
    b, c = meanf.shape
    th = torch.full((b, c), 2 * math.log(h * w)).to(hh[0].device)#torch.from_numpy(2 * np.log(h * w, dtype=float)).unsqueeze(0)
    #th = th.repeat(b, c)
    meanf = torch.sqrt(th) * meanf / 0.6745
    meanf = meanf.unsqueeze(2).unsqueeze(3)
    meanf = meanf.repeat(1, 1, hh[0].shape[-2], hh[0].shape[-1])
    hhf[torch.abs(hhf) < meanf] = 0
    hhf[torch.abs(hhf) >= meanf] = 1
    hhb = torch.cat((hh[4], hh[5], hh[6], hh[7]), dim=1)
    absb = torch.abs(hhb)
    meanb, _ = torch.median(absb.flatten(2), dim=2)
    b, c = meanb.shape
    th = torch.full((b, c), 2 * math.log(h * w)).to(hh[0].device)  # torch.from_numpy(2 * np.log(h * w, dtype=float)).unsqueeze(0)
    #th = th.repeat(b, c)
    meanb = torch.sqrt(th) * meanb / 0.6745
    meanb = meanb.unsqueeze(2).unsqueeze(3)
    meanb = meanb.repeat(1, 1, hh[4].shape[-2], hh[4].shape[-1])
    hhb[torch.abs(hhb) < meanb] = 0
    hhb[torch.abs(hhb) >= meanb] = 1
    mask.append(hhf[:, 0, :, :].unsqueeze(1))
    mask.append(hhf[:, 1, :, :].unsqueeze(1))
    mask.append(hhf[:, 2, :, :].unsqueeze(1))
    mask.append(hhf[:, 3, :, :].unsqueeze(1))
    mask.append(hhb[:, 0, :, :].unsqueeze(1))
    mask.append(hhb[:, 1, :, :].unsqueeze(1))
    mask.append(hhb[:, 2, :, :].unsqueeze(1))
    mask.append(hhb[:, 3, :, :].unsqueeze(1))
    # hhb = torch.cat((hh[4], hh[5], hh[6], hh[7]), dim=1)
    # absb = torch.abs(hhb)
    # meanb = torch.mean(absb, dim=1)

    return mask


def peaks_mask_torch(hh, winsz=7):
    pad = winsz // 2
    stride = pad
    mask_all = []
    for i in range(8):
        x = hh[i].squeeze(1)
        b, h_x, w_x = x.shape
        mask = torch.full((b, h_x, w_x), 0.0).to(x.device)
        x_h = torch.abs(x)
        _, index= torch.nn.functional.max_pool1d(x_h, kernel_size=winsz, padding=pad, stride=stride, return_indices=True)
        bs, h, w = index.shape
        bs_idx = torch.arange(0, bs).view(bs, 1)[:, None, :].repeat(1, h, w)
        h_idx = torch.arange(0, h).view(h, 1)[None, :, :].repeat(bs, 1, 1)
        mask[bs_idx, h_idx, index] = 1

        x_w = torch.abs(x.permute((0, 2, 1)))
        _, index = torch.nn.functional.max_pool1d(x_w, kernel_size=winsz, padding=pad, stride=stride, return_indices=True)
        bs, h, w = index.shape
        bs_idx = torch.arange(0, bs).view(bs, 1)[:, None, :].repeat(1, h, w)
        h_idx = torch.arange(0, h).view(h, 1)[None, :, :].repeat(bs, 1, 1)
        mask = mask.permute((0, 2, 1))
        mask[bs_idx, h_idx, index] = 1
        mask = mask.permute((0, 2, 1))
        mask_all.append(mask.float().unsqueeze(1))
    return mask_all


class SGM(nn.Module):
    def __init__(self, ):
        super(SGM, self).__init__()

        wave_pad = "zero"
        padding = "reflection"
        self.wt = ContourDec(3)


        self.sigmoid = nn.Sigmoid()

    def forward(self, gray):
        self.hh = []
        self.mask = []
        gray = torch.mean(gray, dim=1, keepdim=True)
        gray_ll0, h0 = self.wt(gray)
        self.hh.append(h0)
        self.mask.append(peaks_mask_torch(h0))
        gray_ll1, h1 = self.wt(gray_ll0)
        self.hh.append(h1)
        self.mask.append(peaks_mask_torch(h1))
        gray_ll2, h2 = self.wt(gray_ll1)
        self.hh.append(h2)
        self.mask.append(peaks_mask_torch(h2))
        gray_ll3, h3 = self.wt(gray_ll2)
        self.hh.append(h3)
        self.mask.append(peaks_mask_torch(h3))

        return self.hh, self.mask