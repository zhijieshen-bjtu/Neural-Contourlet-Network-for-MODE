# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from utils.contourlet_torch import DepthToSpace
from utils.contourlet_torch import ContourRec
import torch.nn.functional as F


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class EncoderFusion(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, outch):
        super(EncoderFusion, self).__init__()
        self.iwt = ContourRec()

        self.conv = nn.Sequential(
            nn.Conv2d(outch + 1, outch, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )

    def forward(self, x, hh):
        b, c, h, w = x.shape
        xlo = torch.full((b, 1, h, w), 0).to(x.device)
        edge = self.iwt([xlo, hh])
        edge = F.interpolate(edge, size=[h, w], mode='bilinear', align_corners=True)
        x = torch.cat((x, torch.abs(edge)), dim=1)
        x = self.conv(x)
        # hhf = torch.cat((hh[0], hh[1], hh[2], hh[3]), dim=1)
        # hhf = F.interpolate(hhf, size=[h, w], mode='bilinear', align_corners=True)
        # hhf = DepthToSpace(w_factor=1)(hhf)
        # hhb = torch.cat((hh[4], hh[5], hh[6], hh[7]), dim=1)
        # hhb = F.interpolate(hhb, size=[h, w], mode='bilinear', align_corners=True)
        # hhb = DepthToSpace(h_factor=1)(hhb)
        # x = self.conv(torch.cat((x, hhf, hhb), dim=1))
        # x = torch.nn.functional.normalize(x)
        return x


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers=50, pretrained=False, num_input_images=1, normalize_input=False):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize_input = normalize_input

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.ef1 = EncoderFusion(64)
        self.ef2 = EncoderFusion(256)
        self.ef3 = EncoderFusion(512)
        self.ef4 = EncoderFusion(1024)

    def forward(self, input_image, hh):
        self.features = []
        # x = (input_image - 0.45) / 0.225
        x = input_image
        if self.normalize_input:
            for t, m, s in zip(x, self.mean, self.std):
                t.sub(m).div(s)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        # self.features.append(self.encoder.relu(x))#128 256
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))#64 128
        # self.features.append(self.encoder.layer2(self.features[-1]))#32 64
        # self.features.append(self.encoder.layer3(self.features[-1]))#16 32
        # self.features.append(self.encoder.layer4(self.features[-1]))#8 16 2048

        # sourse
        self.features.append(self.encoder.relu(x))  # 128 256
        self.features.append(self.ef2(self.encoder.layer1(self.encoder.maxpool(self.features[-1])), hh[1]))  # 64 128
        self.features.append(self.ef3(self.encoder.layer2(self.features[-1]), hh[2]))  # 32 64
        self.features.append(self.ef4(self.encoder.layer3(self.features[-1]), hh[3]))  # 16 32
        self.features.append(self.encoder.layer4(self.features[-1]))  # 8 16 2048

        return self.features
