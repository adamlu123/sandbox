import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch.utils.data
from torch import nn, optim
from torch.distributions import Uniform, Normal
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pickle as pkl


class TetrodeSelectNet(nn.Module):
    def __init__(self, input_dim, output_dim, add_bias=False, alternative_sampler=FlowAlternative):
        super(TetrodeSelectNet, self).__init__()
        self.tetrode_group = [3, 1, 8, 4, 6, 1, 4, 3, 1, 5, 7, 1, 1, 1]
        num_tetrodes = len(self.tetrode_group)
        assert sum(self.tetrode_group) == input_dim
        self.input_dim = input_dim
        self.add_bias = add_bias

        self.output_dim = output_dim
        if add_bias:
            self.bias = nn.Parameter(torch.ones(output_dim))
            self.bias.data.uniform_(-0.1, 0.1)

        self.alternative_sampler = alternative_sampler(p=input_dim*output_dim)  # (46 * 4)


    def forward(self, x):
        rep=32
        self.z_group, self.qz_group = self.z_sampler(repeat=rep, logalpha=self.logalpha_group.view(1,-1))  # z.shape = (repeat, batch, p)  qz.shape=(p)
        self.z, self.qz = self.z_group.squeeze(), self.qz_group.squeeze()
        # self.z, self.qz = expand_z(self.z_group, self.qz_goup, self.tetrode_group)

        self.theta, self.logdet, gaussians = self.alternative_sampler.sample(n=rep)
        self.logq = normal_log_pdf(gaussians,
                                   torch.zeros(self.input_dim*self.output_dim).cuda(),
                                   torch.zeros(self.input_dim*self.output_dim).cuda())
        self.Theta = (self.theta.view(-1, 46, self.output_dim).squeeze() * self.z).permute(1,0)  # (16, 46, 4) -> (46 ,4)

        if self.add_bias:
            out = (x.matmul(self.Theta) + self.bias)
        else:
            out = x.matmul(self.Theta)
        return out.mean(dim=1, keepdim=True)  # (168, repeat) -> (168)



class OdorClassification(nn.Module):
    def __init__(self, input_dim, output_dim, SelectNet=TetrodeSelectNet):
        super(OdorClassification, self).__init__()
        self.SelectNet = SelectNet(input_dim, output_dim, add_bias=True)
        self.upper_layer1 = nn.Linear(1, 4)

    def forward(self, x):
        x = x.mean(dim=2)  # (n, 46 ,25)
        x = self.SelectNet(x.squeeze())
        x = F.relu(x)
        x = self.upper_layer1(x)
        return x