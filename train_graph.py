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
        print(out.shape)
        print(out)
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


# def train(optimizer, criterion, data_loader):
#     for data in data_loader:
#         optimizer.zero_grad()
#         x, y = data
#         if is_use_gpu:
#             x, y = x.cuda(), y.cuda()
#         out = model(data.x.cuda(), data.edge_index.cuda())
#         loss = criterion(out, data.y)

#         loss.backward()
#         optimizer.step()

# def train_model(model, data_loader):
#     criterion = torch.nn.CrossEntropyLoss()
#     # or use the
#     optimizer = AdamW(model.parameters(),
#                       lr=learning_rate,
#                       weight_decay=weight_decay)

#     for epoch in range(epoch_num):
#         # initialize loss and accuracy
#         train_loss = 0.
#         train_acc = 0.
#         train_precision = 0.
#         train_recall = 0.
#         train()

#         # train on batches
#         for _, data in enumerate(data_loader):
#             # get batch
#             x, y = data
#             y = get_triu_items(y)
#             if is_use_gpu:
#                 x, y = x.cuda(), y.cuda()
#             # forward
#             z, y_hat = model(x)
#             # backward
#             optimizer.zero_grad()
#             # loss = criterion(y_hat, y) + F.mse_loss(y_hat, y) * 100
#             loss = criterion(y_hat, y)
#             loss.backward()
#             optimizer.step()
#             # accumulate loss and accuracy
#             train_loss += loss
#             # train_acc += (abs(y - y_hat) < delta).float().mean()
#             # true_y_hat = F.relu(1 - y_hat / contrasive_loss_m)
#             zeros = torch.zeros(y.shape)
#             if is_use_gpu:
#                 zeros = zeros.cuda()
#             true_y_hat = torch.maximum(zeros,
#                                        -(y_hat / potential_loss_l - 1)**2 + 1)
#             # precision
#             cor_matrix = (np.abs(true_y_hat.cpu().detach().numpy()) >
#                           0.).astype(float)
#             intr_num = np.count_nonzero(cor_matrix)
#             true_pos = np.count_nonzero(np.multiply(cor_matrix, y.cpu()))
#             if intr_num != 0:
#                 train_precision = true_pos / intr_num
#             ground_truth_intr_num = np.count_nonzero(y.cpu())
#             train_recall = true_pos / ground_truth_intr_num

#         # get loss and accuracy of this epoch
#         loader_step = len(data_loader)
#         train_loss = train_loss / loader_step
#         train_acc = train_acc / loader_step
#         min_loss = min(min_loss, train_loss)
#         # print training stats
#         if epoch == 0 or (epoch + 1) % 10 == 0:
#             print(
#                 f'--- Epoch: {epoch+1:4d}, Loss: {train_loss:.6f}, Interations: {intr_num:6d}, True Pos :{true_pos:6d}, Precision: {train_precision:.2%}, Recall: {train_recall:.2%}'
#             )

#         # print some data for debug
#         if (epoch + 1) == epoch_num:
#             # print('z', z)
#             print('y_hat', y_hat)
#             print('true_y_hat', true_y_hat)
#             print('y', y)

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
