from time import time

import utils.input_data as input_data
from utils.functions import *
import utils.args as args
from utils.node_classification import node_classification
from utils.node_cluster import node_cluster
from models.hgib import HGIB

torch.cuda.empty_cache()        # 清理缓存

datasets = ['IMDB', 'ACM', 'DBLP']      # 按序依次对三个数据集进行实验
for dataset in datasets:
    torch.cuda.empty_cache()
    args.dataset = dataset
    All_data = input_data.load_data(args.dataset, 0.2)  # Reading Data
    Edges = All_data[0]  # Loading edges, for ACM are PA, AP, PC, CP respectively
    Features = All_data[1]  # Loading features
    valid_node = All_data[3]  # The index of nodes for training
    valid_target = All_data[4]  # The labels of nodes for training

    # ACM meta-path: PAP, PCP
    if args.dataset == 'ACM':
        Graph_1 = graphAugment(generate_meta_path(Edges[0], Edges[1]))
        Graph_2 = graphAugment(generate_meta_path(Edges[2], Edges[3]))
        args.n_hidden1 = 512        # 中间层维度
        args.n_hidden2 = 128        # 嵌入层维度
        args.patience = 30          # Loss连续上涨最多次数
        args.learning_rate = 1e-5   # 学习率
    # IMDB meta-path: MDM, MAM
    if args.dataset == 'IMDB':
        Graph_1 = graphAugment(generate_meta_path(Edges[0], Edges[1]))
        Graph_2 = graphAugment(generate_meta_path(Edges[2], Edges[3]))
        args.n_hidden1 = 512
        args.n_hidden2 = 128
        args.patience = 20
        args.learning_rate = 1e-4
    # DBLP meta-path: APA, APCPA
    if args.dataset == 'DBLP':
        Graph_1 = graphAugment(generate_meta_path(Edges[1], Edges[0]))
        Graph_2 = graphAugment(generate_meta_path((generate_meta_path(generate_meta_path(Edges[1], Edges[2]), Edges[3])), Edges[0]))
        args.n_hidden1 = 256
        args.n_hidden2 = 128
        args.patience = 30
        args.learning_rate = 1e-4
    # Get the graph weight based on their number of links
    g1_w, g2_w = graphWeight(Graph_1, Graph_2)      # 计算两条元路径的权重，也可以手动设置
    print('@@@ The meta-path weights are {:.3f} and {:.3f} respectively.'.format(g1_w, g2_w))
    acc = []            # 统计多次实验的分类精度
    clust_acc = []      # 统计多次实验的聚类精度
    # epoch_num = []    # 统计多次实验的迭代次数
    epoch_time = []     # 统计多次试验的运行时间
    ratio = 0.2         # 分类任务中训练集占比

    for i in range(20):     # 在同一数据集下，连续做20次试验
        print('-------------------Round ' + str(i+1) + '-------------------')
        start_time = time()
        # Heterogeneous Graph Information Bottleneck
        model = HGIB(Graph_1, Graph_2, Features)
        model.cuda()
        # Model effect with random initialization 先得出不使用HGIB模型进行训练的情况下，所得到的表示
        init_embeds, _, _ = model.embed(Features, g1_w, g2_w)
        # Training model 正式对HGIB模型进行训练
        embeds, v1, v2, epoch = train_model(model, Features, g1_w, g2_w)
        if i == 0:      # 记录每次实验得到的表示，准备存入pt文件
            embeds_each_iter = torch.unsqueeze(embeds, dim=0)
        else:
            embeds_each_iter = torch.cat((embeds_each_iter, torch.unsqueeze(embeds, dim=0)), dim=0)
        middle1_time = time()
        # epoch_num.append(epoch)
        epoch_time.append(middle1_time - start_time)        # 输出模型训练所用的时间
        print('@@@ Time spent in model training: {:.1f}s'.format(middle1_time - start_time))

        # Node Classifing 节点分类任务
        train_node, train_target, test_node, test_target = input_data.divide_data(All_data[5], ratio)       # 按照ratio的大小随机分割训练集和测试集
        print('@@@ Using {:.0f}% of data for training...'.format(ratio * 100))
        # print('@@@ View 1 for node classifing...')
        # _, _ = node_classification(v1, train_node, valid_node, test_node, train_target, valid_target, test_target)
        # print('@@@ View 2 for node classifing...')
        # _, _ = node_classification(v2, train_node, valid_node, test_node, train_target, valid_target, test_target)
        print('@@@ HGIB embedding for node classifing...')
        acc.append(node_classification(embeds, train_node, valid_node, test_node, train_target, valid_target, test_target))
        middle2_time = time()       # 输出分类所用时间
        print('@@@ Time spent in node classifing: {:.1f}s'.format(middle2_time-middle1_time))

        # Node Clustering 节点聚类任务
        clust_acc.append(node_cluster(embeds[test_node, :], test_target, 'KMeans'))     # 收集每次聚类所得的精度
        end_time = time()
        print('@@@ Time spent in node clustering: {:.1f}s'.format(end_time - middle2_time))
        print("------------------Divide Line------------------")



    #Calculating average score of downstream task
    acc = torch.tensor(acc)
    acc_best_index = ((acc.sum(1)).sort(descending=True)).indices[0:5]
    acc_list_per_ratio = acc[acc_best_index]
    print('@@@ With {:.0f}% ratio of training data, Micro F1: {:.3f}%, Macro F1: {:.3f}%'.format(ratio * 100, acc_list_per_ratio.mean(0)[0], acc_list_per_ratio.mean(0)[1]))
    embeds_each_iter = embeds_each_iter[acc_best_index].detach().cpu().numpy()
    torch.save(embeds_each_iter, 'test_' + dataset + '.pt')
    # epoch_num = torch.tensor(epoch_num).type(torch.float32)
    epoch_time = torch.tensor(epoch_time).type(torch.float32)
    print('@@@ Epoch time about average: {:.1f}s, max: {:.1f}s, min: {:.1f}s'.format(epoch_time.mean(), epoch_time.max(), epoch_time.min()))

    clust_acc = torch.tensor(clust_acc)
    clust_acc_best_index = ((clust_acc.sum(1)).sort(descending=True)).indices[0:5]
    clust_acc = clust_acc[clust_acc_best_index]
    print('Node Clustering results: Micro F1: {:.3f}%, Macro F1: {:.3f}%'.format((clust_acc.mean(0))[0], (clust_acc.mean(0))[1]))
    print('------------------{} data test completed------------------'.format(dataset))
    torch.cuda.empty_cache()