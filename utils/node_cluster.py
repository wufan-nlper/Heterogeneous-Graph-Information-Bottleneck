import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from munkres import Munkres     # 匈牙利算法

def node_cluster(embeds, labels, Algorithm='KMeans'):       # 对所得点的表示进行聚类，默认算法是kmeans
    embeds, labels = embeds.detach().cpu().numpy(), labels.detach().cpu().numpy()
    if Algorithm == 'KMeans':
        from sklearn.cluster import KMeans
        print('Clustering nodes with algorithm KMeans...')
        num_class = np.max(labels).item() + 1
        kmeans = KMeans(n_clusters=num_class, random_state=0).fit(embeds)
        preds = kmeans.predict(embeds)
        preds = fix_preds(preds, labels)
    if Algorithm == 'MeanShift':
        from sklearn.cluster import MeanShift
        print('Clustering nodes with algorithm MeanShift...')
        meanshift = MeanShift().fit(embeds)
        preds = meanshift.predict(embeds)
        preds = fix_preds(preds, labels)
    if Algorithm == 'Spectral':
        from sklearn.cluster import SpectralClustering
        print('Clustering nodes with algorithm SpectralClustering...')
        num_class = np.max(labels).item() + 1
        spectral = SpectralClustering(n_clusters=num_class, random_state=0).fit(embeds)
        preds = spectral.predict(embeds)
        preds = fix_preds(preds, labels)
    accuracy = accuracy_score(labels, preds) * 100
    micro_f1 = f1_score(labels, preds, average='micro') * 100
    macro_f1 = f1_score(labels, preds, average='macro') * 100
    ARI = adjusted_rand_score(labels, preds) * 100
    NMI = normalized_mutual_info_score(labels, preds) * 100
    print('Acc: {:.3f}%, Micro F1: {:.3f}%, Macro F1: {:.3f}%, ARI: {:.3f}%, NMI: {:.3f}%'.format(accuracy, micro_f1, macro_f1, ARI, NMI))
    return accuracy, micro_f1, macro_f1, ARI, NMI       # 聚类算法的五个指标


def fix_preds(preds, labels):       # 对聚类所得的标签进行修正
    m = Munkres()
    num_class = np.max(labels).item() + 1
    label_type_1 = list(set(labels))
    label_type_2 = list(set(preds))
    cost = np.zeros((num_class, num_class), dtype=int)
    for i, c1 in enumerate(label_type_1):
        mps = [i1 for i1, e1 in enumerate(labels) if e1 == c1]
        for j, c2 in enumerate(label_type_2):
            mps_d = [i1 for i1 in mps if preds[i1] == c2]
            cost[i][j] = len(mps_d)
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    actual_preds = np.zeros(preds.size, dtype=int)
    for i, c in enumerate(label_type_1):
        c2 = label_type_2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(preds) if elm == c2]
        actual_preds[ai] = c
    return actual_preds
# 暂未完成
class Deep_Cluster(nn.Module):
    def __init__(self, embeds, labels):
        super(Deep_Cluster, self).__init__()
        self.embeds = embeds
        self.labels = labels
        self.num_class = np.max(labels).item() + 1
        self.hidden = embeds.shape[1]
        self.cluster_centers = nn.Parameter(torch.tensor(self.pretrain(), dtype=torch.float).cuda())

    def forward(self):

        return

    def pretrain(self):
        from sklearn.cluster import KMeans
        embeds, labels = self.embeds.detach().cpu().numpy(), self.labels.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_class, random_state=0).fit(embeds)
        return kmeans.cluster_centers_

