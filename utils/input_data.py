import pickle
import torch
import random
import numpy as np
def load_data(dataset, ratio, CUDA_flag=True):      # 数据集名称，训练集占比，是否使用CUDA
    with open('data/' + dataset + '/node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)      #点特征
    with open('data/' + dataset + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)      # 四段元路径 如:PA,AP,PS,SP
    with open('data/' + dataset + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)     # 标签

    labels_ = np.concatenate((labels[0], labels[2]), axis=0)
    node_num = labels_.shape[0]
    labels_ = labels_[random.sample(range(node_num), node_num)]
    test_labels = labels_[int(node_num*ratio):]

    if CUDA_flag == True:
        A = []
        for i, edge in enumerate(edges):
            A.append(torch.from_numpy(edge.todense()).type(torch.FloatTensor).cuda())
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor).cuda()
        valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor).cuda()
        valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor).cuda()
    else:
        A = []
        for i, edge in enumerate(edges):
            A.append(torch.from_numpy(edge.todense()).type(torch.FloatTensor))
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
        valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor)
        valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor)

    num_classes = torch.max(torch.from_numpy(test_labels[:, 1]).type(torch.LongTensor)).item() + 1
    return A, node_features, num_classes, valid_node, valid_target, labels_
# 以上分别对应：
# 四种对应的邻接矩阵（torch.Tensor）组成的List，全不对称
# 节点特征（（8994，1902）.torch.Tensor），P点3025个（0，3024），A点5912个（3025，8936），S点57个（8937，8993），有5个P点没有指向任何S点
# 节点种类数（3）
# 训练集节点索引（共600个）
# 训练集节点对应的标签 0：21，1：129，2：450
# 验证集（300个）
# 验证集标签 0：0，1：56，2：244
# 测试集（2125个）
# 测试集标签 0：694，1：670，2：761，占比分别是32.7%，31.5%，35.8%

#随机分割数据
def divide_data(data, ratio, CUDA_flag=True):
    length = data.shape[0]
    data = data[random.sample(range(length), length)]
    train_data = data[0:int(ratio*length)]
    test_data = data[int(ratio*length):]

    if CUDA_flag == True:
        train_node = torch.from_numpy(train_data[:, 0]).type(torch.LongTensor).cuda()
        train_target = torch.from_numpy(train_data[:, 1]).type(torch.LongTensor).cuda()
        test_node = torch.from_numpy(test_data[:, 0]).type(torch.LongTensor).cuda()
        test_target = torch.from_numpy(test_data[:, 1]).type(torch.LongTensor).cuda()
    else:
        train_node = torch.from_numpy(train_data[:, 0]).type(torch.LongTensor)
        train_target = torch.from_numpy(train_data[:, 1]).type(torch.LongTensor)
        test_node = torch.from_numpy(test_data[:, 0]).type(torch.LongTensor)
        test_target = torch.from_numpy(test_data[:, 1]).type(torch.LongTensor)

    return train_node, train_target, test_node, test_target
