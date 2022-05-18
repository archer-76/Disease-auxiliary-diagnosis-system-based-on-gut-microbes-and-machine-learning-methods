import torch
import torch.nn as nn
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
import scipy.sparse as sp
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(1, 8, 3, 1, 1),
                                    nn.MaxPool1d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv1d(8, 16, 3, 1, 1),
                                    nn.MaxPool1d(2, 2))
        self.layer3 = nn.Sequential(nn.Conv1d(16, 1, 3, 1, 1),
                                    nn.MaxPool1d(2, 2), nn.Flatten())
        self.layer4 = nn.Sequential(nn.Linear(int(in_dim / 8), hidden_dim),
                                    nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class DiffPool(nn.Module):
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

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

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
        # embedding = nn.Embedding(x.shape[1], 64)
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


class GNN(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=False):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

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
class GraphBatchedDiffPool(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_channels,
                 num_nodes,
                 is_final=False):
        super(GraphBatchedDiffPool, self).__init__()
        self.embed = GNN(num_features, hidden_channels, hidden_channels)
        self.pool = GNN(num_features, hidden_channels, num_nodes)
        self.link_pred_loss = 0
        self.entropy_loss = 0

    def forward(self,
                x: torch.tensor,
                edge_index,
                adj: torch.Tensor,
                nothing,
                mask=None):
        z_l = self.embed(x, adj)
        s_l = F.softmax(self.pool(x, adj), dim=-1)
        x, adj, l, e = dense_diff_pool(z_l, adj, s_l, mask)
        self.link_pred_loss = l
        self.entropy_loss = e
        if (x.dim() == 3):
            x = x.reshape((x.shape[1], -1))
        assert (x.dim() == 2)
        return x, adj


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self,
                 in_features,
                 out_features,
                 dropout=0.6,
                 alpha=0.2,
                 device='cpu',
                 concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.device = 'cpu'
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(
            h,
            self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_features) + ' -> ' + str(self.out_features) + ')'


class BatchedGat(nn.Module):
    def __init__(self, infeat, outfeat, device='cpu', dropout=0.6, nheads=4):
        super(BatchedGat, self).__init__()
        self.dropout = dropout
        self.device = device
        self.attentions = [
            GraphAttentionLayer(infeat,
                                32,
                                dropout=self.dropout,
                                device=self.device) for _ in range(nheads)
        ]
        self.attentions = []
        for _ in range(nheads):
            self.attentions.append(
                GraphAttentionLayer(infeat,
                                    32,
                                    dropout=self.dropout,
                                    device=self.device))
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(32 * nheads,
                                           outfeat,
                                           dropout=self.dropout,
                                           concat=False)

    def forward(self, x, adj):
        if (x.dim() == 3):
            x = x.reshape((x.shape[1], x.shape[2]))
            adj = adj[0]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        if (x.dim() == 2):
            x = x.reshape((1, x.shape[0], x.shape[1]))
            adj = adj.reshape((1, adj.shape[0], adj.shape[1]))
        x = F.normalize(x, dim=2, p=2)
        x = F.relu(x)
        return x


class BatchedSpGat(nn.Module):
    def __init__(self, infeat, outfeat, device='cpu', use_bn=True):
        super().__init__()
        # 使用4头注意力
        self.device = device
        self.gat1 = GATConv(infeat, 32, 4, dropout=0.6).to(self.device)
        # 多头注意力是把向量拼接
        self.gat2 = GATConv(128, outfeat, 1, dropout=0.6).to(self.device)

    # 问题在于现在传入的是adj，应该把edge_index与adj都传进来，但是只用其中一个
    def forward(self, x, edge_index: torch.Tensor):
        x = x.reshape((x.shape[1], -1))
        print('edg', edge_index.shape)
        # if (not edge_index.is_cuda):
        #     print('x', x)
        #     edge_index.to(self.device)
        #     print('edge', edge_index.shape, edge_index)
        assert (x.is_cuda == edge_index.is_cuda)
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        if (x.dim() == 2):
            x = x.reshape((1, x.shape[0], x.shape[1]))
            print('x.shape', x.shape)
        assert (x.dim() == 3)
        x = F.normalize(x, dim=2, p=2)
        x = F.relu(x)
        return x


class GatDiffPool(nn.Module):
    def __init__(self,
                 nfeat,
                 nnext,
                 nout,
                 device='cpu',
                 is_final=False,
                 link_pred=False):
        super(GatDiffPool, self).__init__()
        self.link_pred = link_pred
        self.device = device
        self.is_final = is_final
        self.embed = BatchedGraphSAGE(nfeat, nout, device=self.device)
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, device=self.device)
        self.log = {}
        self.link_pred_loss = 0.
        self.entropy_loss = 0.

    def forward(self, x, adj, mask=None, log=False):
        z_l = self.embed(x, adj)
        s_l = self.assign_mat(x, adj)
        xnext, anext, self.link_pred_loss, self.entropy_loss = dense_diff_pool(
            z_l, adj, s_l)
        return xnext, anext


class BatchedGraphSAGE(nn.Module):
    def __init__(self,
                 infeat,
                 outfeat,
                 device='cpu',
                 use_bn=True,
                 mean=False,
                 add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.device = device
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj, mask=None):
        if self.add_self:
            adj = adj + torch.eye(adj.size(0)).to(self.device)

        if self.mean:
            adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(h_k.size(1)).to(self.device)
            if (h_k.shape[0] != 1):
                h_k = self.bn(h_k)
        if mask is not None:
            h_k = h_k * mask.unsqueeze(2).expand_as(h_k)
        return h_k


class BatchedDiffPool(nn.Module):
    def __init__(self,
                 nfeat,
                 nnext,
                 nout,
                 device='cpu',
                 is_final=False,
                 link_pred=False):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.device = device
        self.is_final = is_final
        self.embed = BatchedGraphSAGE(nfeat,
                                      nout,
                                      device=self.device,
                                      use_bn=True)
        self.assign_mat = BatchedGraphSAGE(nfeat,
                                           nnext,
                                           device=self.device,
                                           use_bn=True)
        self.log = {}
        self.link_pred_loss = 0.
        self.entropy_loss = 0.

    def forward(self, x, adj, mask=None, log=False):
        z_l = self.embed(x, adj)
        s_l = F.softmax(self.assign_mat(x, adj), dim=-1)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
        assert (anext.shape[0] == 1)
        tmp_adj = anext.cpu().detach().reshape(anext.shape[1], -1)
        tmp_coo = sp.coo_matrix(tmp_adj)
        values = tmp_coo.data
        indices = np.vstack((tmp_coo.row, tmp_coo.col))
        i = torch.LongTensor(indices)
        v = torch.LongTensor(values)
        coo = torch.sparse_coo_tensor(i, v, tmp_coo.shape)
        coo_next = i.to(self.device)
        if self.link_pred:
            # TODO: Masking padded s_l
            self.link_pred_loss = (adj -
                                   s_l.matmul(s_l.transpose(-1, -2))).norm(
                                       dim=(1, 2))
            self.entropy_loss = torch.distributions.Categorical(
                probs=s_l).entropy()
            if mask is not None:
                self.entropy_loss = self.entropy_loss * mask.expand_as(
                    self.entropy_loss)
            self.entropy_loss = self.entropy_loss.sum(-1)
            # adj_matrix 是邻接矩阵
        return xnext, anext, coo_next


class Classifier(nn.Module):
    def __init__(self, input_shape=30, n_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(input_shape, 50), nn.ReLU(),
                                        nn.Linear(50, n_classes))

    def forward(self, x):
        return self.classifier(x)


class BatchedModel(nn.Module):
    def __init__(self,
                 pool_size,
                 input_shape,
                 n_classes,
                 device,
                 link_pred=False):
        super().__init__()
        self.input_shape = input_shape
        self.link_pred = link_pred
        self.device = device
        self.layers = nn.ModuleList([
            BatchedGraphSAGE(input_shape, 30, device=self.device),
            BatchedGraphSAGE(30, 30, device=self.device),
            BatchedDiffPool(30, pool_size, 30, self.device, self.link_pred),
            BatchedGraphSAGE(30, 30, device=self.device),
            BatchedGraphSAGE(30, 30, device=self.device),
            BatchedDiffPool(30, 1, 30, self.device)
        ])
        # self.gatlayers = nn.ModuleList([
        #     BatchedGat(input_shape, 30, device=self.device),
        #     BatchedGat(30, 30, device=self.device),
        #     GatDiffPool(30, pool_size, 30, self.device, self.link_pred),
        #     BatchedGat(30, 30, device=self.device),
        #     BatchedGat(30, 30, device=self.device),
        #     GatDiffPool(30, 1, 30, self.device)
        # ])
        self.classifier = Classifier(30, n_classes)

    # def forward(self, x, adj, mask):
    #     adj = self.add_self_loop(adj)
    #     for i, layer in enumerate(self.gatlayers):
    #         if isinstance(layer, BatchedGat):
    #             # print('gat.shape', x.shape, 'type x', type(x),
    #             #       'type edge_index', type(edge_index))

    #             if mask is not None:
    #                 if mask.shape[1] == x.shape[1]:
    #                     x = layer(x, adj, mask)
    #             else:
    #                 x = layer(x, adj)
    #         elif isinstance(layer, GatDiffPool):
    #             # TODO: Fix if condition
    #             # print('diff.shape', x.shape, 'type x', type(x),
    #             #       'type edge_index', type(edge_index))
    #             if mask is not None:
    #                 if mask.shape[1] == x.shape[1]:
    #                     x, adj = layer(x, adj, mask)
    #             else:
    #                 x, adj = layer(x, adj)
    #         # print('x', x.shape)

    #     # x = x * mask
    #     # readout_x = x.sum(dim=1)
    #     readout_x = self.classifier(x)

    #     return readout_x

    def forward(self, x, adj, mask):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchedGraphSAGE):
                if mask is not None:
                    if mask.shape[1] == x.shape[1]:
                        x = layer(x, adj, mask)
                else:
                    x = layer(x, adj)
            elif isinstance(layer, BatchedDiffPool):
                # TODO: Fix if condition
                if mask is not None:
                    if mask.shape[1] == x.shape[1]:
                        x, adj = layer(x, adj, mask)
                else:
                    x, adj, _ = layer(x, adj)
            # print('x', x.shape)

        # x = x * mask
        # readout_x = x.sum(dim=1)
        readout_x = self.classifier(x)

        return readout_x

    def loss(self, output, labels):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        if self.link_pred:
            # 在计算池化层时，这部分应该被单独计算
            for layer in self.layers:
                if isinstance(layer, BatchedDiffPool):
                    loss = loss + layer.link_pred_loss + layer.entropy_loss
        return loss

    def add_self_loop(self, adj):
        if adj.dim() == 3:
            for i in range(adj.shape[0]):
                adj[i] = adj[i] + torch.eye(adj[i].shape[0]).to(self.device)

        elif adj.dim() == 2:
            adj = adj + torch.eye(adj.shape[0])
        else:
            print('adj shape', adj.shape)
        return adj
