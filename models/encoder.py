import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import utils.args as args
from models.gcn import GCN
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_feats):
        super(Encoder, self).__init__()
        self.base_gcn = GCN(in_feats, args.n_hidden1, args.activation)
        self.mean_gcn = GCN(args.n_hidden1, args.n_hidden2, activation=lambda x:x)

    def forward(self, g, x):
        h = self.base_gcn(g, x)
        mean = self.mean_gcn(g, h)
        return mean