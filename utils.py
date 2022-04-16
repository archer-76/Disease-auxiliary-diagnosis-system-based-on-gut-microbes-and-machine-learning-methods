from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
# load dataset
from torch_geometric.datasets import TUDataset
from minepy import cstats, pstats
from scipy.stats import spearmanr


def get_benchmark_dataset(dataset_name='ENZYMES'):
    dataset = TUDataset('data/TUDataset', name=dataset_name)
    dataset = dataset.shuffle()
    datalist = []
    for i in range(len(dataset)):
        data = dataset[i]
        data = Data(x=data.x, y=data.y, edge_index=data.edge_index)
        datalist.append(data)
    return datalist


def get_dataset(smpl_path: str, muti_target: bool = False, threshold=0.4):
    """Generate dataset from csv files.

    Args:
        smpl_path (str):
            Path of samples file.

        muti_target(Boolean):
            use multiple target or not
        y:
            Adjenct matrix of samples with shape of samples_num, samples_num
            y is generated by MENA, involving tons of diseased and healthy samples
    Returns:
        List: list of Data(x, edge_index) every data is a sub_graph.
    """
    data_list = []
    disease_list = []
    samples_df = pd.read_csv(smpl_path,
                             header=None,
                             dtype={
                                 "disease": str,
                                 'n': float,
                                 'obesity': float,
                                 't2d': float,
                                 'ibd': float,
                                 'adenoma': float,
                                 'cirrhosis': float
                             })
    # keep only the rows with at least n non-zero values
    y_list = samples_df.iloc[0, 1:].to_numpy()
    drop_thresh = int(0.15 * y_list.shape[0])
    print(drop_thresh)
    samples_df = samples_df.replace(0, np.nan)
    samples_df = samples_df.dropna(thresh=drop_thresh)
    samples_df = samples_df.replace(np.nan, 0)

    samples = samples_df.iloc[1:, 1:].to_numpy(dtype=np.float32)

    le = LabelEncoder()
    le = le.fit(['adenoma', 'cirrhosis', 'ibd', 'n', 'obesity', 't2d', 'wt2d'])
    # le = le.fit(['leaness', 'obesity'])
    y_list = le.transform(y_list).reshape(1, -1)
    # 按照首字母排序，n的下标为3，第四个，如果不是多分类任务，设置n为0，其他为1
    if (not muti_target):
        print(np.max(y_list), np.min(y_list))
        y_list = np.where(y_list == 3, 0, 1)
        print(f'positive sample: {np.count_nonzero(y_list) }',
              f'out of: {len(y_list[0])} ',
              f'positive rate: {np.count_nonzero(y_list)/len(y_list[0])}')

    for i in range(np.max(y_list) + 1):
        mask = y_list == i
        mask = mask.reshape((mask.shape[1]))
        col = samples[:, mask]
        disease_list.append(col)

    giant_edge_index, giant_adj = construct_adj(disease_list,
                                                names='pcc mic  spm',
                                                methods=(mic, spm, pcc),
                                                score_thresh=threshold)
    print(
        f'{np.count_nonzero(giant_adj)} pred out of {samples.shape[0]**2} true'
    )
    print('giant_edge_index.shape', giant_edge_index.shape,
          np.count_nonzero(giant_adj))
    for i in range(samples.shape[1]):
        x = samples[:, i]
        y = y_list[:, i]
        adj, edge_index = get_edge_index(x, giant_adj)
        x = x[x != 0]
        x = x.reshape((x.shape[0], 1))
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=int)
        assert (x.dim() == 2)
        data = Data(x=x, y=y, adj=adj, edge_index=edge_index)
        data_list.append(data)

    return data_list


# 在测试后，cutoff为0.62，0.64，0.68的时候卡方检验泊松分布转化完成
# p-value分别为0.05,0.01,0.001，此x必须是t2d数据集
# 如果更换数据集，请重新使用MENA软件测试http://ieg4.rccc.ou.edu/mena/
# 对于giant_matrix多次检验结果分别是0.66，0.68，0.69
def construct_adj(xs: list,
                  names: str,
                  methods: tuple[function, ...],
                  score_thresh=0.5):
    """construct the enormous adjenct matrix.

    Args:
        xs: diseases, otus, samples
        methods: calculate correlation score
        score_thresh: beyond which, cor_score will remain, otherwise set to zero 
    Returns:
        only one mixed giant adj and correspond edge index
    """
    # pre process
    print('loading giant')
    for x in xs:
        print('non zero ', np.count_nonzero(x), ' out of total ',
              np.count_nonzero(np.where(x == 0, np.max(x), x)))
        x /= np.min(np.where(x == 0, np.max(x), x))
        x = np.where(x == 0, (np.e)**(-1), x)
        x = np.log(x)
        x = np.where(x == -1, 0.01, x)

        assert (not np.isnan(np.max(x)))
        # print('xmax', np.max(x), 'xavg', np.average(x))

    print('pre done')
    cor_matrix = np.zeros((xs[0].shape[0], xs[0].shape[0]))
    # each disease
    for j, x in enumerate(xs):
        cor_matrixs = np.zeros((len(methods), len(x), len(x)))
        # each method
        for i in range(len(methods)):
            # 忘了取绝对值
            cor_matrixs[i] = np.abs(methods[i](x))
            np.fill_diagonal(cor_matrixs[i], 1)
            cor_matrixs[i] = np.where(np.abs(cor_matrixs[i] >= score_thresh),
                                      cor_matrixs[i], 0)
            np.savetxt(f'data/matrix/cor_matrix{j} {i}.txt',
                       cor_matrixs[i],
                       fmt='%.2f')
            assert (not np.isnan(np.max(cor_matrixs[i])))
            # 这个结果没有起到筛选阈值的效果
            np.multiply(cor_matrixs[i], np.abs(cor_matrixs[i]) >= score_thresh)

        for i in range(len(cor_matrixs)):
            cor_matrix = cor_matrix + cor_matrixs[i]
    # self loop and symmetric
    cor_matrix = cor_matrix + cor_matrix.T + np.eye(cor_matrix.shape[0])
    # 0 and 1 only
    cor_matrix = np.where(cor_matrix != 0, 1, 0)
    np.savetxt(f'data/matrix/mixed adj of {names}.txt', cor_matrix, fmt='%.2f')
    # edge index
    adj_sparse = coo_matrix(cor_matrix)
    adj_indices = np.vstack((adj_sparse.row, adj_sparse.col))
    cor_edg = torch.LongTensor(adj_indices)
    return cor_edg, cor_matrix


def get_edge_index(x: np.ndarray,
                   giant_adj: np.ndarray,
                   method: str = "index"):
    """return new adjenct matrix of x by looking up in giant_adj, the giant adjenct matrix.

    Args:
        x:
            a row of the OTU table, represent to a single sample, a Kranker.
        giant_adj:
            Adjenct matrix of samples with shape of samples_num, samples_num
            giant_adj is generate by MENA or MIC, involved tons of diseased and healthy samples
        method:
            if index:
                generate the edge_index by index

    Returns:
        edge_index: shape of (2,interactions)
    """
    if method == 'index':
        mask = x != 0
        tmp = giant_adj[mask]
        adj = tmp[:, mask]
        adj = adj + adj.T + np.eye(adj.shape[0])
        adj_sparse = coo_matrix(adj)
        adj_indices = np.vstack((adj_sparse.row, adj_sparse.col))
        edge_index = torch.LongTensor(adj_indices)
    return adj, edge_index


def mic(x):
    mic_score, tic_score = pstats(x, alpha=0.6, c=15, est="mic_e")
    mic_adj = np.zeros((x.shape[0], x.shape[0]))
    tic_adj = np.zeros((x.shape[0], x.shape[0]))
    ind = np.triu_indices(x.shape[0], 1)
    mic_adj[ind] = mic_score
    tic_adj[ind] = tic_score
    return mic_adj


def pcc(x):
    return np.corrcoef(x)


def spm(x):

    return spearmanr(x, axis=1)[0]


def get_dataset_for_MENA(smpl_path: str, flag=True):
    """pre_process the csv files.

    Args:
        smpl_path (str): Path of samples file.

    Returns:
        no returns, directly change the file.
    """
    samples_df = pd.read_csv('./data/samples.csv')
    if flag == True:
        otus = samples_df.iloc[1:, :].to_numpy()
        otu_figures = otus[:, 1:]
        otu_figures = np.where(otu_figures < np.nanmin(otu_figures) * 100,
                               np.NaN, otu_figures)
        otu_index = np.arange(int(otu_figures.shape[1] / 2))
        np.random.shuffle(otu_index)
        print(otu_index.shape)
        otu_index = otu_index.reshape((otu_index.shape[0], -1))
        otu_figures = otu_figures[:, otu_index]
        print(otu_figures.shape)
        otu_figures = otu_figures.reshape((otu_figures.shape[0], -1))
        otu_figures *= (1. / np.nanmin(otu_figures))
        otus = np.hstack(((1 + np.asarray(range(otus.shape[0]))).reshape(
            (-1, 1)), otu_figures + 0.01))
        df = pd.DataFrame(otus,
                          columns=['name'] +
                          [f's{i+1}' for i in range(otus.shape[1] - 1)])

    else:
        df = samples_df.dropna(thresh=15)
    df.to_csv('./data/giant_matrix.csv', index=False)
