import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import manifold
from utils.node_cluster import fix_preds
'''对所得的表示进行可视化'''
dataset = 'acm'

#file_path = 'test_DGI_' + dataset + '.pt'
file_path = 'test_v2_' + dataset + '.pt'
t = torch.load(file_path)

label_path = 'data/' + dataset + '/labels.pkl'
y = pickle.load(open(label_path, 'rb'))
y = np.concatenate((y[0], y[1], y[2]), axis=0)
t = t[:, 0:(y[:, 0].max() + 1), :]
n_class = y[:, 1].max() + 1
y = y[np.lexsort(y[:,::-1].T)]
num = 0     # 表示用第几组数据，因为之前分类任务选了最好的5个，因此num的取值为0~4

kmeans = KMeans(n_clusters=n_class, random_state=0).fit(t[num])
y_pred = kmeans.predict(t[num])
y_pred = fix_preds(y_pred, y[:, 1])
'''
spectral = SpectralClustering(n_clusters=n_class, random_state=0).fit(t[num])
y_pred = spectral.labels_
y_pred = fix_preds(y_pred, y[:, 1])
'''
# t-SNE算法做降维并可视化
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X = tsne.fit_transform(t[num])
x_min, x_max = X.min(0), X.max(0)
X_norm = (X - x_min) / (x_max - x_min)
# 聚类所分配的标签
plt.figure()
for i in range(X_norm.shape[0]):
    #plt.text(X_norm[i, 0], X_norm[i, 1], str(y_pred[i]), color=plt.cm.Set1(y_pred[i]), fontdict={'weight': 'bold', 'size': 9})
    plt.text(X_norm[i, 0], X_norm[i, 1], '.', color=plt.cm.Set1(y_pred[i]), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()
# 原标签
plt.figure()
for i, j in enumerate(y[:, 0]):
    #plt.text(X_norm[j, 0], X_norm[j, 1], str(y[i][1]), color=plt.cm.Set1(y[i][1]), fontdict={'weight': 'bold', 'size': 9})
    plt.text(X_norm[j, 0], X_norm[j, 1], '.', color=plt.cm.Set1(y[i][1]), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()