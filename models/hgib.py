import torch
import torch.nn as nn
from torch.optim import Adam

import utils.args as args
from models.encoder import Encoder
from models.mi_estimator import MIEstimator

class HGIB(nn.Module):
    def __init__(self, g1, g2, fts):
        super(HGIB, self).__init__()
        self.g1 = g1
        self.g2 = g2
        self.encoder_v1 = Encoder(fts.shape[1])
        #self.encoder_v2 = Encoder()
        self.encoder_v2 = self.encoder_v1
        self.mi_estimator = MIEstimator(args.n_hidden2, args.n_hidden2)
        self.kl_estimator_1 = MIEstimator(args.n_hidden2, args.n_hidden2)
        self.kl_estimator_2 = MIEstimator(args.n_hidden2, args.n_hidden2)
        self.opt = Adam(self.parameters(), lr=args.learning_rate)

    def forward(self, x1, x2):
        z1 = self.encoder_v1(self.g1, x1)   # view1的embedding，其pooling后的结果为v1，view2同理
        v1 = torch.mean(z1, dim=0)
        v1 = v1.expand_as(z1)
        z2 = self.encoder_v2(self.g2, x2)
        v2 = torch.mean(z2, dim=0)
        v2 = v2.expand_as(z2)

        mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        mi_gradient = mi_gradient.mean()
        mi_estimation = mi_estimation.mean()

        skl_v1_z2, _ = self.kl_estimator_1(v1, z2)
        skl_v2_z1, _ = self.kl_estimator_2(v2, z1)
        skl = skl_v1_z2 + skl_v2_z1
        skl = skl.mean()

        self.loss = -mi_gradient + args.beta * skl

        return mi_estimation

    def compute_loss(self):
        return self.loss

    def train_step(self):
        self.train()
        loss = self.loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.eval()

    def embed(self, fts, z1w, z2w):
        z1 = self.encoder_v1(self.g1, fts)
        z2 = self.encoder_v2(self.g2, fts)
        return z1*z1w+z2*z2w, z1, z2