import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
# import torch_geometric.nn.DataParallel as dp
from math import ceil
import torch.optim as optim
from tqdm import tqdm
from utils import *
from models import *
from sklearn import metrics
# DEBUG
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# device = torch.device('cpu')

model_path = './models/gmlp_model.pth'
smpl_path = './data/samples.csv'

batch_size = 16
learning_rate = 1e-2
weight_decay = 1e-3
epoch_num = 500
train_radio = 0.5
valid_radio = 0.5 - train_radio / 2

is_use_gpu = torch.cuda.is_available()
is_save_model = True


def unbatched_train(train_loader, model):
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


def unbatched_test(test_loader, model):
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


def batched_train(loader, model):
    model.train()
    epoch_loss = 0.
    iter_precise = 0.
    timer = 0
    for _, batch in enumerate(loader):
        # 这里解决了一个问题，之前是放在batch之外的
        loss = 0.
        for data in batch.to_data_list():
            x, edge_index, y = data.x, data.edge_index, data.y
            dense_y = torch.sparse_coo_tensor(
                edge_index, torch.ones(edge_index.shape[1]),
                torch.Size([x.shape[0], x.shape[0]]))
            adj = dense_y.to_dense()
            if adj.dim() == 2:
                x = x.reshape((1, x.shape[0], x.shape[1]))
                adj = adj.reshape((1, adj.shape[0], adj.shape[1]))
            # adj = adj @ adj
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            out = model(x, adj, None)
            out = out.reshape((out.shape[0], -1))
            loss += model.loss(out, y)
            timer += 1
            iter_precise = iter_precise + metrics.precision_score(
                y.cpu().detach(),
                out.cpu().detach().argmax(dim=1),
                average='micro')
            # iter_precise = iter_precise + metrics.precision_score(
            #     y.cpu().detach(), y.cpu().detach(), average='micro')
        loss.backward()
        # 梯度截断防止梯度爆炸导致过拟合，类似于dropout
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    return epoch_loss / len(loader), iter_precise / timer


def batched_test(loader, model, testing: bool = False):
    model.eval()
    iter_precise = 0.
    timer = 0
    for _, batch in enumerate(loader):
        for data in batch.to_data_list():
            x, edge_index, y = data.x, data.edge_index, data.y
            dense_y = torch.sparse_coo_tensor(
                edge_index, torch.ones(edge_index.shape[1]),
                torch.Size([x.shape[0], x.shape[0]]))
            adj = dense_y.to_dense()
            if adj.dim() == 2:
                x = x.reshape((1, x.shape[0], x.shape[1]))
                adj = adj.reshape((1, adj.shape[0], adj.shape[1]))
            # adj = adj @ adj
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            out = model(x, adj, None)
            out = out.reshape((out.shape[0], -1))
            timer += 1
            if testing:
                print('out', out.argmax(dim=1), 'y', y)
            iter_precise = iter_precise + metrics.precision_score(
                y.cpu().detach(),
                out.cpu().detach().argmax(dim=1),
                average='micro')

    return iter_precise / timer


if __name__ == '__main__':
    data_list = get_dataset(smpl_path, muti_target=True, threshold=0.32)
    # data_list = get_benchmark_dataset('PROTEINS')
    train_list, test_list, validation_list = random_split(
        data_list,
        (int(train_radio * len(data_list)), int(valid_radio * len(data_list)),
         (len(data_list) - int(train_radio * len(data_list)) -
          int(valid_radio * len(data_list)))))

    train_loader = DataLoader(train_list, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_list, batch_size)
    test_loader = DataLoader(test_list, batch_size)

    valid_precise_list = np.zeros((1))
    in_feature = train_list[0].x.shape[1]
    classes = max([item.y for item in data_list]) + 1
    maxmum_nodes = max([item.x.shape[0] for item in data_list])
    pool_size = ceil(maxmum_nodes * 0.25)
    model = BatchedModel(pool_size,
                         input_shape=in_feature,
                         n_classes=int(classes),
                         device=device).to(device)

    optimizer = optim.Adam(model.parameters())
    for e in tqdm(range(epoch_num)):
        train_loss, train_precise = batched_train(train_loader, model)
        valid_precise = batched_test(validation_loader, model)
        valid_precise_list = np.vstack((valid_precise_list, valid_precise))
        tqdm.write(
            f'''Epoch:{e+1} \t train_precise:{train_precise:.3f} \t valid_precise:{valid_precise:.3f} \t train_loss:{train_loss:.3f}'''
        )
    # model.load_state_dict(torch.load(model_path))
    test_precise = batched_test(test_loader, model, True)
    print('test_precise', test_precise, 'max_valid_precise',
          np.max(valid_precise_list))
    # save last model
    if is_save_model:
        torch.save(model.state_dict(), model_path)
    #
    # train_model(model, data_loader)
