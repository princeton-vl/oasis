import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool
import os

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class EdgeNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, **kwargs):
        super(EdgeNet, self).__init__()
        self.pre = nn.Sequential(
            #Conv(3, 64, 7, 2, bn=bn),
            Conv(3, 64, 7, 1, bn=bn),
            Conv(64, 128, bn=bn),
            #Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        self.features = nn.ModuleList( [
        nn.Sequential(
            Hourglass(5, inp_dim, bn, increase), # Orig 4
            Conv(inp_dim, inp_dim, 3, bn=False),
            Conv(inp_dim, inp_dim, 3, bn=False)
        ) for i in range(nstack)] )

        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )

        self.nstack = nstack

    def forward(self, imgs):
        #x = imgs.permute(0, 3, 1, 2) # TODO
        x = imgs
        x = self.pre(x)
        preds_list = []
        for i in range(self.nstack):
            feature = self.features[i](x)
            preds_list.append(self.outs[i](feature))
            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds_list[-1]) + self.merge_features[i](feature)
        for i in range(len(self.features)):
            preds_list[i] = torch.sigmoid(preds_list[i])
        return preds_list
