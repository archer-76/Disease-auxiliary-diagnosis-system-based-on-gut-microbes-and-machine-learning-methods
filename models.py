import torch
from torch.nn import Linear
import torch.nn.functional as F
from utils import *
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm as BN1d
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
from torch.optim import Adam, AdamW
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from math import ceil
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DiffPool(torch.nn.Module):
    def __init__(self,
                 num_features,
                 hidden_channels,
                 num_classes,
                 max_nodes=400,
                 normalize=False):
        super(DiffPool, self).__init__()
        num_nodes = ceil(0.25 * max_nodes)

        self.gnn1_pool = GNN(num_features, hidden_channels, num_nodes)
        self.gnn1_embed = GNN(num_features, hidden_channels, hidden_channels)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(hidden_channels, hidden_channels, num_nodes)
        self.gnn2_embed = GNN(
            hidden_channels,
            hidden_channels,
            hidden_channels,
        )

        self.gnn3_embed = GNN(
            hidden_channels,
            hidden_channels,
            hidden_channels,
        )

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.tensor, adj: torch.Tensor, mask=None):
        '''
            args:
                input: 
                    should be a single graph's batch, 
                    which means batch consists of nodes but not graphs
                output: 
                    out value for the criterion, in this 2-class classification task
                    value should be like ( float , float ) with shape (1,2)

        
        '''
        # 用词向量的方式，扩充节点特征从1维到64维
        # embedding = torch.nn.Embedding(x.shape[1], 64)
        # print(x.shape)
        # x = embedding(x)
        # print(x.shape)
        # 从 edge_index 生成adj，因为 adj 不能被直接传参，他被视为占了两个参数
        # reversed_edge_index = torch.vstack(
        #     (edge_index[1, :], edge_index[0, :]))
        # edge_index = torch.hstack((edge_index, reversed_edge_index))
        # dense_y = torch.sparse_coo_tensor(
        #     edge_index.detach().cpu(),
        #     torch.ones(edge_index.detach().cpu().shape[1]),
        #     (x.detach().cpu().shape[0], x.detach().cpu().shape[0]))
        # dense_y = dense_y.to_dense()
        # adj = dense_y.fill_diagonal_(1.).to(device)
        # s.shape should be B*N*C
        s = x
        s = self.gnn1_pool(s, adj, mask)
        # s = self.gnn1_pool(x, adj, mask)
        # s.shape should be B*N*F
        x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0
        s = x
        s = self.gnn2_pool(s, adj, mask)
        # s = self.gnn2_pool(x, adj, mask)
        x = self.gnn2_embed(x, adj, mask)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj, mask)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x, l1 + l2, e1 + e2

    # return F.log_softmax(x, dim=-1), l1, e1


class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=False):
        super(GNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(BN1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(BN1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(BN1d(out_channels))
        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        for step in range(len(self.convs)):
            tmp = F.relu(self.convs[step](x, adj, mask=None))
            tmp = tmp.reshape((tmp.shape[1], -1))
            x = self.bns[step](tmp)
        return x

    # this forward used for mutiple graphs
    # def forward(self, x, adj, useless):
    #     for step in range(len(self.convs)):
    #         tmp = F.relu(self.convs[step](x, adj))
    #         x_shape0 = x.shape[0]
    #         # torch.Size([16, 290, 64])
    #         # 这里的维度维度要求应该同上，而不是290
    #         # bn1d是作用在290这个维度上的，想办法让他作用在64上
    #         for graph in range(tmp.shape[0]):
    #             tmp_x = self.bns[step](tmp[graph])
    #             x = torch.vstack((tmp_x, tmp_x))
    #     assert (x.shape[0] == x_shape0)
    #     return x


class BatchedDiffPool(torch.nn.Module):
    def __init__(self,
                 num_features,
                 hidden_channels,
                 num_nodes,
                 is_final=False):
        super(BatchedDiffPool, self).__init__()
        self.embed = GNN(num_features, hidden_channels, hidden_channels)
        self.pool = GNN(num_features, hidden_channels, num_nodes)
        self.link_pred_loss = 0
        self.entropy_loss = 0

    def forward(self, x: torch.tensor, adj: torch.Tensor, mask=None):
        z_l = self.embed(x, adj)
        s_l = F.softmax(self.pool(x, adj), dim=-1)
        x, adj, l, e = dense_diff_pool(z_l, adj, s_l, mask)
        self.link_pred_loss = l
        self.entropy_loss = e
        if (x.dim() == 3):
            x = x.reshape((x.shape[1], -1))
        assert (x.dim() == 2)
        return x, adj


class Classifier(torch.nn.Module):
    def __init__(self, input_shape=30, n_classes=2):
        super().__init__()
        self.classifier = torch.nn.Sequential(torch.nn.Linear(input_shape, 50),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(50, n_classes))

    def forward(self, x):
        return self.classifier(x)


class BatchedModel(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 pool_size,
                 input_shape,
                 n_classes,
                 link_pred=False):
        super().__init__()
        self.input_shape = input_shape
        self.link_pred = link_pred
        self.device = device
        self.layers = torch.nn.ModuleList([
            GNN(input_shape, hidden_channels, 30),
            GNN(30, hidden_channels, 30),
            BatchedDiffPool(30, 30, pool_size),
            GNN(30, hidden_channels, 30),
            GNN(30, hidden_channels, 30),
            BatchedDiffPool(30, 30, 1, is_final=True)
        ])
        self.classifier = Classifier(30, n_classes)

    def forward(self, x, adj, mask):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GNN):
                if mask.shape[1] == x.shape[1]:
                    x = layer(x, adj, mask)
                else:
                    x = layer(x, adj)
            elif isinstance(layer, BatchedDiffPool):
                # TODO: Fix if condition
                if mask.shape[1] == x.shape[1]:
                    x, adj = layer(x, adj, mask)
                else:
                    x, adj = layer(x, adj)
            # print('x', x.shape)

        # x = x * mask
        # readout_x = x.sum(dim=1)
        readout_x = self.classifier(x)

        return readout_x

    def loss(self, output, labels):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        if self.link_pred:
            # 在计算池化层时，这部分应该被单独计算
            for layer in self.layers:
                if isinstance(layer, BatchedDiffPool):
                    loss = loss + layer.link_pred_loss.mean(
                    ) + layer.entropy_loss.mean()
        return loss
