# Heterogeneous Graph Information Bottleneck
This repository is the implementation of Heterogeneous Graph Information Bottleneck(HGIB)
> Liang Yang, Fan Wu, Zichen Zheng, Bingxin Niu, Junhua Gu, Chuan Wang, Xiaochun Cao and Yuanfang Guo. "Heterogeneous Graph Information Bottleneck". IJCAI-21
## Overview
Here we provide the implementation of Heterogeneous Graph Information Bottleneck(HGIB) in Pytorch, 
along with a minimal execution example. The repository is organised as follows:
- `data/` contains the necessary dataset files for ACM, DBLP and IMDB;
- `models/` contains the implementation of the HGIB and its necessary parts;
- `utils/` contains:
    * parameter settings in `args.py`;
    * some preprocessing subroutines `functions.py`;
    * transforming data files into appropriate tensors `input_data.py`;
    * downstream machine learning tasks `node_classification.py`, `node_cluster.py`.
Finally, `main.py` puts all of the above together and may be used to execute a full training run. 
When training step completes, you can use `visualization.py` to obtain the visual distribution of
node representations. 

## Dependencies
- `pytorch==3.6.10`
- `tqdm==4.36.1`
- `numpy==1.18.2`
- `scikit-learn==0.22.1`
- `munkres==1.1.4`
