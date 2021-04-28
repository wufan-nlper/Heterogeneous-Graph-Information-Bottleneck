from __future__ import division

import torch
import torch.nn as nn
from tqdm import tqdm
from models.logreg import LogReg
import utils.args as args


def graphAugment(g):        # 生成D^1/2AD^1/2矩阵
    x, y = g.shape
    #diagl = torch.zeros(x, y)       #draft3
    diagl = torch.zeros(x, y).cuda()       #draft
    for i in range(x):
        if torch.sum(g[i]) != 0.:
            diagl[i][i] = torch.pow(torch.sum(g[i]), -0.5)
    return torch.mm(torch.mm(diagl, g), diagl)

def generate_meta_path(x, y):       # 使得元路径矩阵中大于1的元素等于1
    z = x.mm(y)
    z_ones = torch.ones_like(z)
    z = torch.where(z > 1.0, z_ones, z)
    return z

def graphWeight(x, y):      # 按元路径中链接的数量作为元路径在embedding中的比重
    sum_x = x.sum()
    sum_y = y.sum()
    x_weight = sum_x / (sum_x+sum_y)
    y_weight = sum_y / (sum_x+sum_y)
    return x_weight, y_weight

def train_model(model, Features, g1_w, g2_w):
    print('Training model with dataset:', args.dataset)
    best = 10000  # Top limit of loss
    stop_count = 0  # When loss > best, stop_count will increase
    for epoch in tqdm(range(args.num_epoch_limit)):
        mi = model(Features, Features)  # The mutual information between two meta-paths
        loss = model.compute_loss()
        if loss < best:
            stop_count = 0
            best = loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_hml.pkl')
        else:
            stop_count += 1
        if stop_count == args.patience:
            print('\nEarly Stopped!')
            break
        model.train_step()
    print('Loading {}th epoch'.format(best_epoch + 1))
    model.load_state_dict(torch.load('best_hml.pkl'))
    embeds, view1, view2 = model.embed(Features, g1_w, g2_w)  # Ignore the single embedding for each view
    return embeds, view1, view2, best_epoch

def find_epoch(hid_units, nb_classes, train_embs, train_lbls, test_embs, test_lbls):
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0.00001)
    xent = nn.CrossEntropyLoss()
    log.cuda()

    epoch_flag = 0
    epoch_win = 0
    best_acc = torch.zeros(1).cuda()

    for e in range(20000):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward(retain_graph=True)
        opt.step()

        if (e + 1) % 10 == 0:
            log.eval()
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            if acc >= best_acc:
                epoch_flag = e + 1
                best_acc = acc
                epoch_win = 0
            else:
                epoch_win += 1
            if epoch_win == 10:
                break
    return epoch_flag
