from turtle import forward
from unittest import loader
from matplotlib.pyplot import get
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
        self.bns.append(BN1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(BN1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(BN1d(out_channels))
        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, adj, useless):
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
        s = self.gnn1_pool(x, adj, mask)
        # s.shape should be B*N*F
        x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0

        # s = self.gnn2_pool(x, adj, mask)
        # x = self.gnn2_embed(x, adj, mask)

        # x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj, mask)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        # return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
        return F.log_softmax(x, dim=-1), l1, e1


# # model = GCN(hidden_channels=64)
model = DiffPool(num_features=1, hidden_channels=64, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()
# criterion = F.nll_loss()
# output, data.y.view(-1)


def train(train_loader):
    model.train()
    train_loss = 0.
    # 一个batch读取出来16张图
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        x, adj, y = batch.x, batch.adj, batch.y
        # 对16个图都分别训练：
        loss = torch.tensor([0.]).to(device)
        for j in range(x.shape[0]):
            data = Data(x=x[j].reshape((x[j].shape[0], -1)),
                        y=y[j].reshape((-1)),
                        adj=adj[j].reshape((adj[j].shape[1], -1)))
            optimizer.zero_grad()
            out, _, _ = model(data.x, data.adj)
            out = out.reshape((1, out.shape[0]))
            assert (out.dim() == 2)
            # loss += criterion(out, y[j])
            loss += F.nll_loss(out, y[j])
        # 这里的batchsize不对，留待后续处理
        loss /= batch_size
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
