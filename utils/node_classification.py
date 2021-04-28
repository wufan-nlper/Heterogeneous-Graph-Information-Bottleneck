import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import f1_score

import utils.args as args
from models.logreg import LogReg
from utils.functions import find_epoch

def node_classification(embeds, train_node, valid_node, test_node, train_target, valid_target, test_target):
    train_embeds = embeds[train_node, :]
    valid_embeds = embeds[valid_node, :]
    test_embeds = embeds[test_node, :]

    num_class = torch.max(train_target).item() + 1
    node_dim = args.n_hidden2
    xent = nn.CrossEntropyLoss()

    log = LogReg(node_dim, num_class)
    log.cuda()
    print('Searching for property number of epoch...')
    n_of_log = find_epoch(node_dim, num_class, train_embeds, train_target, test_embeds, test_target)
    print('Node Classify Epoches: ', n_of_log)
    print('Classifing now...')
    opt = Adam(log.parameters(), lr=0.001, weight_decay=0.00001)
    for _ in tqdm(range(n_of_log)):
        log.train()
        opt.zero_grad()
        logits = log(train_embeds)
        cls_loss = xent(logits, train_target)
        cls_loss.backward(retain_graph=True)
        opt.step()
    logits = log(test_embeds)
    preds = torch.argmax(logits, dim=1)
    preds, test_target = preds.cpu().numpy(), test_target.cpu().numpy()
    micro_f1_test = f1_score(preds, test_target, average='micro') * 100
    macro_f1_test = f1_score(preds, test_target, average='macro') * 100
    print('Test set index:')
    print('Micro F1: {:.3f}%, Macro F1: {:.3f}%'.format(micro_f1_test, macro_f1_test))

    logits = log(train_embeds)
    preds = torch.argmax(logits, dim=1)
    preds, train_target = preds.cpu().numpy(), train_target.cpu().numpy()
    micro_f1 = f1_score(preds, train_target, average='micro') * 100
    macro_f1 = f1_score(preds, train_target, average='macro') * 100
    print('Train set index:')
    print('Micro F1: {:.3f}%, Macro F1: {:.3f}%'.format(micro_f1, macro_f1))

    logits = log(valid_embeds)
    preds = torch.argmax(logits, dim=1)
    preds, valid_target = preds.cpu().numpy(), valid_target.cpu().numpy()
    micro_f1 = f1_score(preds, valid_target, average='micro') * 100
    macro_f1 = f1_score(preds, valid_target, average='macro') * 100
    print('Valid set index:')
    print('Micro F1: {:.3f}%, Macro F1: {:.3f}%'.format(micro_f1, macro_f1))

    return micro_f1_test, macro_f1_test