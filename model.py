import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import sys
from utils.resnet import ResnetEncoder
from utils.decoder import DecoderContourlet
from utils.SGM import SGM
from utils.contourlet_torch import ContourRec
import numpy as np
sys.path.append('../../../')


class ContouletNet(nn.Module):
    def __init__(self, ):
        super(ContouletNet, self).__init__()

        print("Building model ", end="")

        self.sgm = SGM()
        self.encoder = ResnetEncoder(pretrained=True)
        self.decoder = DecoderContourlet(enc_features=self.encoder.num_ch_enc, decoder_width=0.5)

    def forward(self, x, threshold=-1):
        hh, mask = self.sgm(x)
        x = self.encoder(x, hh)
        depth = self.decoder(x, mask)
        return depth