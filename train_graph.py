from unittest import loader
from matplotlib.pyplot import get
import torch
from torch.nn import Linear
import torch.nn.functional as F
from utils import *
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
from torch.optim import Adam, AdamW
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from math import ceil

smpl_path = './data/t2d.csv'
a = './MENA network/t2d.csv'
b = './MENA network/t2d_1.csv'
batch_size = 16
learning_rate = 1e-4
weight_decay = 1e-3
epoch_num = 200

is_use_gpu = torch.cuda.is_available()
is_save_model = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))
        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, adj, useless):
        for step in range(len(self.convs)):
            tmp = F.relu(self.convs[step](x, adj))
            print(tmp.shape)
            x = self.bns[step](tmp)
            print('adj', adj)
            print('useless', useless)
        return x


class DiffPool(torch.nn.Module):
    def __init__(self,
                 num_features,
                 hidden_channels,
                 num_classes,
                 max_nodes=250,
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
        print(adj.shape)
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x, l1 + l2, e1 + e2


# # model = GCN(hidden_channels=64)
model = DiffPool(num_features=1, hidden_channels=64, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train(train_loader):
    model.train()
    train_loss = 0.
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.adj)
        loss = criterion(out, data.y)
        loss.backward()
        train_loss += loss
        optimizer.step()
    return train_loss / len(train_loader)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # 批遍历测试集数据集。
        data = data.to(device)
        out = model(data.x, data.adj, data.batch)  # 一次前向传播
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        correct += int((pred == data.y).sum())  # 检查真实标签
    return correct / len(loader.dataset)


if __name__ == '__main__':
    train_data_list = get_dataset(smpl_path)
    train_loader = DenseDataLoader(train_data_list, batch_size, shuffle=True)
    for step, batch in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print('batch:', batch)
        print(f'adj in the current batch: {batch.adj.shape}')
    for epoch in range(1, epoch_num + 1):
        loss = train(train_loader)
        train_acc = test(train_loader)
        print(
            # f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}'
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, loss: {loss:.4f}'
        )
        # test_acc = test(test_loader)
    # model.load_state_dict(torch.load(model_path))
    # train_model(model, data_loader)
