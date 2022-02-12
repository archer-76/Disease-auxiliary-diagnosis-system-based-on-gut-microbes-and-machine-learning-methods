from unittest import loader
from matplotlib.pyplot import get
import torch
from torch.nn import Linear
import torch.nn.functional as F
from utils import *
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim import Adam, AdamW
from torch_geometric.datasets import TUDataset

smpl_path = './data/obesity.csv'
a = './MENA network/t2d.csv'
b = './MENA network/t2d_1.csv'
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-3
epoch_num = 200

is_use_gpu = torch.cuda.is_available()
is_save_model = True

# dataset = TUDataset('data/TUDataset', name='MUTAG')
# torch.manual_seed(12345)
# dataset = dataset.shuffle()

# train_dataset = dataset[:150]
# test_dataset = dataset[150:]
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. 获得节点嵌入
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 分类器
        x = F.dropout(x, p=0.5, training=self.training)
        # 有一个权重矩阵，用来转换特征
        x = self.lin(x)
        # x.shape[batch_size, num_classes]
        return x


# # model = GCN(hidden_channels=64)
model = GraphSAGENet(in_channels=1, hidden_channels=64, out_channels=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train(train_loader):
    model.train()

    for data in train_loader:
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # 批遍历测试集数据集。
        out = model(data.x, data.edge_index, data.batch)  # 一次前向传播
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        correct += int((pred == data.y).sum())  # 检查真实标签
    return correct / len(loader.dataset)


if __name__ == '__main__':
    train_data_list = get_dataset(smpl_path)
    train_loader = DataLoader(train_data_list, batch_size, shuffle=True)

    # model = GatNet(hidden_channels=64)
    # if is_use_gpu:
    #     model = model.cuda()

    # for step, batch in enumerate(train_loader):
    #     print(f'Step {step + 1}:')
    #     print('=======')
    #     print(f'Number of graphs in the current batch: {batch.num_graphs}')
    #     print(batch)
    #     print()

    for epoch in range(1, epoch_num + 1):
        train(train_loader)
        train_acc = test(train_loader)
        # test_acc = test(test_loader)
        print(
            # f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}'
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')

    # model.load_state_dict(torch.load(model_path))
    # train_model(model, data_loader)
