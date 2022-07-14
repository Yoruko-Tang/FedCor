#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import abc
import math

class FedModule(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def Get_Local_State_Dict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def Load_Local_State_Dict(self,local_dict):
        raise NotImplementedError()



class MLP(nn.Module,FedModule):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.layers = []
        self.relus = []
        self.dropouts = []
        
        if len(dim_hidden)>0:
            self.layers.append(nn.Linear(dim_in, dim_hidden[0]))
            self.relus.append(nn.ReLU())
            self.dropouts.append(nn.Dropout())
            for n in range(len(dim_hidden)-1):
                self.layers.append(nn.Linear(dim_hidden[n],dim_hidden[n+1]))
                self.relus.append(nn.ReLU())
                self.dropouts.append(nn.Dropout())
            self.layers.append(nn.Linear(dim_hidden[-1], dim_out))
        else:
            # logistic regression
            self.layers.append(nn.Linear(dim_in, dim_out))
        

        self.layers = nn.ModuleList(self.layers)
        self.relus = nn.ModuleList(self.relus)
        self.dropouts = nn.ModuleList(self.dropouts)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        for n in range(len(self.relus)):
            x = self.layers[n](x)
            x = self.dropouts[n](x)
            x = self.relus[n](x)
        x = self.layers[-1](x)
        return x


    def Get_Local_State_Dict(self):
        # save local parameters without weights and bias
        sd = self.state_dict()
        for name in list(sd.keys()):
            if 'weight' in name or 'bias' in name:
                sd.pop(name)
        return sd

    def Load_Local_State_Dict(self,local_dict):
        # load local parameters saved by Get_Local_State_Dict()
        sd = self.state_dict()
        sd.update(local_dict)
        self.load_state_dict(sd)


                    

class NaiveCNN(nn.Module,FedModule):
    def __init__(self, args,input_shape = [3,32,32],final_pool=True):
        super(NaiveCNN, self).__init__()
        self.convs = []
        self.fcs = []
        self.final_pool=final_pool
        if len(args.kernel_sizes) < len(args.num_filters):
            exlist = [args.kernel_sizes[-1] for i in range(len(args.num_filters)-len(args.kernel_sizes))]
            args.kernel_sizes.extend(exlist)
        elif len(args.kernel_sizes) > len(args.num_filters):
            exlist = [args.num_filters[-1] for i in range(len(args.kernel_sizes)-len(args.num_filters))]
            args.num_filters.extend(exlist)
        output_shape = np.array(input_shape)
        for ksize in args.kernel_sizes[:-1] if not final_pool else args.kernel_sizes:
            if args.padding:
                pad = ksize//2
                output_shape[1:] = (output_shape[1:]+2*pad-ksize-1)//2+1
            else:
                output_shape[1:] = (output_shape[1:]-ksize-1)//2+1
        if not final_pool:
            if args.padding:
                pad = args.kernel_sizes[-1]//2
                output_shape[1:] = output_shape[1:]+2*pad-args.kernel_sizes[-1]+1
            else:
                output_shape[1:] = output_shape[1:]-args.kernel_sizes[-1]+1
        output_shape[0] = args.num_filters[-1]
        conv_out_length = output_shape[0]*output_shape[1]*output_shape[2]
        
        self.convs.append(nn.Conv2d(input_shape[0], args.num_filters[0], kernel_size=args.kernel_sizes[0],padding = args.kernel_sizes[0]//2 if args.padding else 0))
        for n in range(len(args.num_filters)-1):
            self.convs.append(nn.Conv2d(args.num_filters[n], args.num_filters[n+1], kernel_size=args.kernel_sizes[n+1],padding = args.kernel_sizes[n+1]//2 if args.padding else 0))
        #self.conv2_drop = nn.Dropout2d()
        self.fcs.append(nn.Linear(conv_out_length, args.mlp_layers[0]))
        for n in range(len(args.mlp_layers)-1):
            self.fcs.append(nn.Linear(args.mlp_layers[n], args.mlp_layers[n+1]))
        self.fcs.append(nn.Linear(args.mlp_layers[-1], args.num_classes))
        
        self.convs = nn.ModuleList(self.convs)
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        for n in range(len(self.convs)-1 if not self.final_pool else len(self.convs)):
            x = F.relu(F.max_pool2d(self.convs[n](x), 2))
        if not self.final_pool:
            x = F.relu(self.convs[-1](x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        for n in range(len(self.fcs)-1):
            x = F.relu(self.fcs[n](x))
            #x = F.dropout(x, training=self.training)
        x = self.fcs[-1](x)
        return x
        

    def Get_Local_State_Dict(self):
        sd = self.state_dict()
        for name in list(sd.keys()):
            if 'weight' in name or 'bias' in name:
                sd.pop(name)
        return sd

    def Load_Local_State_Dict(self,local_dict):
        # load local parameters saved by Get_Local_State_Dict()
        sd = self.state_dict()
        sd.update(local_dict)
        self.load_state_dict(sd)

    
class BNCNN(NaiveCNN):
    def __init__(self, args,input_shape = [1,28,28]):
        super(BNCNN, self).__init__(args,input_shape)
        self.bns = []
        for num_filter in args.num_filters:
            self.bns.append(nn.BatchNorm2d(num_filter))
        self.bns = nn.ModuleList(self.bns)
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2))
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2))
        # self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        for n in range(len(self.convs)):
            x = F.relu(F.max_pool2d(self.bns[n](self.convs[n](x)), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        for n in range(len(self.fcs)-1):
            x = F.relu(self.fcs[n](x))
            #x = F.dropout(x, training=self.training)
        x = self.fcs[-1](x)
        return x

# ==================================ResNet==================================
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module,FedModule):

    def __init__(self, depth, num_classes=10):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    def Get_Local_State_Dict(self):
        sd = self.state_dict()
        for name in list(sd.keys()):
            if 'weight' in name or 'bias' in name:
                sd.pop(name)
        return sd

    def Load_Local_State_Dict(self,local_dict):
        # load local parameters saved by Get_Local_State_Dict()
        sd = self.state_dict()
        sd.update(local_dict)
        self.load_state_dict(sd)

class RNN(nn.Module):
    def __init__(self, n_hidden, n_classes,embedding=8,pretrain_emb=False,fc_layer = None):
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.fc_layer = fc_layer
        if not pretrain_emb:
            self.emb = nn.Embedding(n_classes, embedding)
        else:
            self.emb = None
        self.LSTM = nn.LSTM(embedding, n_hidden, 2)
        if fc_layer is not None:
            self.hidden_fc = nn.Linear(n_hidden,fc_layer)
            self.fc = nn.Linear(fc_layer, n_classes)
        else:
            self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, features):
        if self.emb is not None: # [batch,seq_len]
            x = self.emb(features.T) #[seq_len, batch, emb_len]
        else: # [batch, seq_len, emb_len]
            x = torch.transpose(features,0,1) #[seq_len, batch, emb_len]
        x, _ = self.LSTM(x) #[seq_len, batch, n_hidden]
        if self.fc_layer is not None:
            x = self.hidden_fc(x[-1, :, :])#[batch, fc_layer]
            x = self.fc(x) #[batch, n_classes]
        else:
            x = self.fc(x[-1, :, :]) #[batch, n_classes]
        return x

    def Get_Local_State_Dict(self):
        sd = self.state_dict()
        for name in list(sd.keys()):
            if 'weight' in name or 'bias' in name:
                sd.pop(name)
        return sd

    def Load_Local_State_Dict(self,local_dict):
        # load local parameters saved by Get_Local_State_Dict()
        sd = self.state_dict()
        sd.update(local_dict)
        self.load_state_dict(sd)


if __name__ == '__main__':
    rnn = RNN(8, 256, 80)
    sd = rnn.state_dict()
    for name in list(sd.keys()):
        print(name)
    