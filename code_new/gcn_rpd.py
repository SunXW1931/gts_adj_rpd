# -*- coding: utf-8 -*-
import json, os
import math

import numpy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))  # 将不可训练的tensor转换为可以训练的parameter
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        '''
        随机初始化参数
        :return:
        '''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, text, rpd, length):
        hidden = torch.matmul(text, self.weight)
        denom = length
        output = torch.matmul(rpd, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCNModel_rpd(nn.Module):
    def __init__(self, args):
        super(GCNModel_rpd, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(2*args.lstm_dim, 2*args.lstm_dim)
        self.gc2 = GraphConvolution(2*args.lstm_dim, 2*args.lstm_dim)
        self.gc3 = GraphConvolution(2 * args.lstm_dim, 2 * args.lstm_dim)

    def forward(self, lstm_feature, wpp,length, mask):
        inputs = lstm_feature
        rpd = wpp
        x = F.relu(self.gc1(inputs, rpd,length))
        x = F.relu(self.gc2(x, rpd,length))
        x = x * mask.unsqueeze(2).float().expand_as(x)

        output = x
        return output