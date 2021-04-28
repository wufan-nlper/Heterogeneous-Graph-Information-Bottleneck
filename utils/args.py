# 数据设定
import torch.nn.functional as F

n_hidden1 = 512
n_hidden2 = 128
activation = F.relu
patience = 30
num_epoch_limit = 50000
learning_rate = 1e-4
beta = 1e-3