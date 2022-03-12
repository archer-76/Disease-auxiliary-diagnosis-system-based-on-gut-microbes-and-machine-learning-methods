import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
from torch.optim import Adam, AdamW
from torch_geometric.loader import DataLoader
from math import ceil
import torch.optim as optim
from utils import *
from models import *
# DEBUG
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

smpl_path = './data/t2d.csv'
a = './MENA network/t2d.csv'
b = './MENA network/t2d_1.csv'
batch_size = 16
learning_rate = 1e-4
weight_decay = 1e-3
epoch_num = 100

is_use_gpu = torch.cuda.is_available()
is_save_model = True


def train(train_loader, model):
    model.train()
    train_loss = 0.
    # 一个batch读取出来16张图
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        x, adj, y = batch.x, batch.adj, batch.y
        # 对16个图(x.shape[0])都分别训练：
        loss = torch.tensor([0.]).to(device)
        for j in range(x.shape[0]):
            data = Data(x=x[j].reshape((x[j].shape[0], -1)),
                        y=y[j].reshape((-1)),
                        adj=adj[j].reshape((adj[j].shape[1], -1))).to(device)
            # data 为十六个图中的一个
            out = model(data.x, data.adj, mask=np.ones((13, 11)))
            assert (out.dim() == 2)
            loss += model.loss(out, y[j])
            # loss += F.nll_loss(out, y[j])
        # 这里的batchsize不对，留待后续处理
        loss /= x.shape[0]
        loss += 0 * sum([x.sum() for x in model.parameters()])
        loss.backward()
        train_loss += loss
        optimizer.step()
        optimizer.zero_grad()
    return train_loss / len(train_loader)


def test(test_loader, model):
    model.eval()
    correct = 0
    num_graph = 0
    for batch in test_loader:  # 批遍历测试集数据集。
        batch = batch.to(device)
        x, adj, y = batch.x, batch.adj, batch.y
        for j in range(x.shape[0]):
            data = Data(x=x[j].reshape((x[j].shape[0], -1)),
                        y=y[j].reshape((-1)),
                        adj=adj[j].reshape((adj[j].shape[1], -1)))
            # data 为十六个图中的一个
            out = model(data.x, data.adj, mask=np.ones((13, 11)))
            assert (out.dim() == 2)
            pred = out.argmax(dim=-1)  # 使用概率最高的类别
            # print('out', out, 'pred', pred)
            correct += int(pred == data.y)  # 检查真实标签
        num_graph += x.shape[0]
    return correct / num_graph


if __name__ == '__main__':
    data_list = get_dataset(smpl_path)
    random.shuffle(data_list)
    train_list = data_list[:ceil(2 / 3 * len(data_list))]
    test_list = data_list[ceil(2 / 3 * len(data_list)):]
    train_loader = DenseDataLoader(train_list, batch_size, shuffle=True)
    test_loader = DenseDataLoader(data_list, batch_size, shuffle=True)
    maxmum_nodes = 290
    pool_size = ceil(maxmum_nodes * 0.25)
    model = BatchedModel(30, pool_size, 1, 2).to(device)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, epoch_num + 1):
        loss = train(train_loader, model)
        train_acc = test(test_loader, model)
        print(
            # f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}'
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, loss: {loss.item():.4f}'
        )
    # test_acc = test(test_loader)
    # model.load_state_dict(torch.load(model_path))
    # train_model(model, data_loader)

# # model = GCN(hidden_channels=64)
# model = DiffPool(num_features=1, hidden_channels=64, num_classes=2).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()
# criterion = F.nll_loss()
# output, data.y.view(-1)