import os
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import argparse
from functools import reduce


class SiameseNet(nn.Module):
    def __init__(self, args):
        super(SiameseNet, self).__init__()
        self.fc = nn.Conv2d(args.F, args.fc_dimensions, 1, 1)
        self.bn = nn.BatchNorm2d(args.fc_dimension)
        self.ReLU = nn.ReLU()


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class MIML(nn.Module):
    def __init__(self, args):
        super(MIML, self).__init__()
        self.args = args
        self.fc = nn.Conv2d(args.F, args.fc_dimensions, 1, 1)
        self.bn_1 = nn.BatchNorm2d(args.fc_dimensions)
        self.ReLU = nn.ReLU()
        self.sub_concept_layer = nn.Conv2d(args.fc_dimensions, args.K * args.L, 1, 1)
        self.bn_2 = nn.BatchNorm2d(args.K * args.L)
        self.sub_concept_pooling = nn.MaxPool2d((args.K, 1), stride=(1, 1))
        self.softmax = nn.LogSoftmax(dim=-1)
        self.basis_pooling = nn.MaxPool2d((args.M, 1), stride=(1, 1))
        #init_weights(self, self.args.init_type)


    def forward(self, input):
        # input: (batch_size, F, M)
        # labels: (batch_size, L)
        # basis_vector: (batch_size, F, M, 1)
        basis_vector = input.unsqueeze(3)
        # feature_map: (batch_size, fc_dimension, M, 1)
        # print(basis_vector.shape)
        fc = self.fc(basis_vector)
        bn_1 = self.bn_1(fc)
        feature_map = self.ReLU(bn_1)
        #print(feature_map.shape)
        # feature_cube: (batch_size, L, K, M)
        feature_cube = self.ReLU(self.bn_2(self.sub_concept_layer(feature_map))).view(-1, self.args.L, self.args.K, self.args.M)
        #print(feature_cube.shape)
        # sub_concept_pooling: (batch_size, 1, M, L)
        sub_concept_pooling = self.sub_concept_pooling(feature_cube).view(-1, self.args.L, self.args.M).permute(0,2,1).unsqueeze(1)
        #print(sub_concept_pooling.shape)
        # output: (batch_size, L)
        output = self.basis_pooling(sub_concept_pooling).view(-1, self.args.L)
        output = self.softmax(output)
        return output, sub_concept_pooling

