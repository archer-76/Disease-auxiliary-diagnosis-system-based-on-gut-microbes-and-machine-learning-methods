import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

smpl_path = './data/obesity.csv'
a = './MENA network/t2d.csv'


def get_dataset(smpl_path: str) -> tuple:
    """Generate dataset from csv files.

    Args:
        smpl_path (str):
            Path of samples file.
        y:
            Adjenct matrix of samples with shape of samples_num, samples_num
            y is generate by MENA, involved tons of diseased and healthy samples
    Returns:
        List: list of Data(x, edge_index).
    """
    samples_df = pd.read_csv(smpl_path, header=None)
    # keep only the rows with at least n non-zero values
    samples_df = samples_df.replace(0, np.nan)
    samples_df = samples_df.dropna(thresh=10)
    samples_df = samples_df.replace(np.nan, 0)
    y_list = samples_df.iloc[0, 1:].to_numpy()
    le = LabelEncoder()
    le = le.fit(['leaness', 'obesity'])
    y_list = le.transform(y_list).reshape(1, -1)
    samples = samples_df.iloc[1:, 1:].to_numpy(dtype=np.float32)
    assert (y_list.shape[1] == samples.shape[1])
    _, adj = construct_adj(samples)
    data_list = []
    for i in range(samples.shape[1]):
        x = samples[:, i]
        y = 1
        if y_list[:, i] == 0.:
            y = 0
        edge_index = get_edge_index(x, adj, method="index")
        x = x.reshape((x.shape[0], 1))
        x = torch.as_tensor(x, dtype=torch.float32)
        assert (x.dim() == 2)
        data = Data(x=x, y=y, edge_index=edge_index)
        data_list.append(data)

    return data_list


def get_edge_index(x: np.ndarray, y: np.ndarray, method: str):
    """return adjenct matrix of x by looking up in y, the giant adjenct matrix.

    Args:
        x:
            a row of the OTU table, represent to a single sample, a Kranker.
        y:
            Adjenct matrix of samples with shape of samples_num, samples_num
            y is generate by MENA, involved tons of diseased and healthy samples
        method:
            if index:
                generate the edge_index by index

    Returns:
        edge_index: shape of (2,interactions)
    """
    if method == 'index':
        index = np.nonzero(x)
        adj = y[index, index]
        adj_sparse = coo_matrix(adj)
        adj_indices = np.vstack((adj_sparse.row, adj_sparse.col))
        edge_index = torch.LongTensor(adj_indices)
    return edge_index


def get_dataset_for_MENA(smpl_path: str, flag=True):
    """pre_process the csv files.

    Args:
        smpl_path (str): Path of samples file.

    Returns:
        no returns, directly change the file.
    """
    samples_df = pd.read_csv(smpl_path)
    if flag == True:
        otus = samples_df.iloc[:, 1:].to_numpy(dtype=np.float32)
        otu_names = np.char.add('OTU', np.arange(otus.shape[0]).astype(str))
        otus = np.hstack((otu_names[:, None], otus))
        df = pd.DataFrame(otus,
                          columns=['name'] +
                          [f's{i+1}' for i in range(otus.shape[1] - 1)])
    else:
        df = samples_df.dropna(thresh=5)
    df.to_csv(a, index=False)


# 在测试后，cutoff为0.62，0.64，0.68的时候卡方检验泊松分布转化完成
# p-value分别为0.05,0.01,0.001，此x必须是t2d数据集
# 如果更换数据集，请重新使用MENA软件测试
def construct_adj(x: np.ndarray, score_thresh=0.61):
    """construct the enormous adjenct matrix.

    Args:
        x (np.ndarray): the OTU used to calculate corrcoef score.
        score_thresh  : from MENA online website to cut off
    Returns:
        edge_index: shape of (2, num_of_cor_pairs)
        adj : shape of (x.shape[0],x.shape[0])
    """
    x = x * 10
    x = np.where(x == 0, 0.01, x)
    adj = np.corrcoef(x)
    adj = np.multiply((np.abs(adj) > score_thresh), adj)
    np.abs(adj) > score_thresh
    adj_sparse = coo_matrix(adj)
    adj_indices = np.vstack((adj_sparse.row, adj_sparse.col))
    edge_index = torch.LongTensor(adj_indices)
    return edge_index, adj

    # get_dataset_for_MENA(a, False)


train_data_list = get_dataset(smpl_path)
