import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.args as args


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, activation=F.relu, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = activation
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, g, x):
        h = self.fc(x)
        h = torch.mm(g, h)
        h = self.activation(h)
        return h