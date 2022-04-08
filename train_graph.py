import random
from click import Parameter
import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
from torch.optim import Adam, AdamW
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from math import ceil
import torch.optim as optim
from tqdm import tqdm
from utils import *
from models import *
# DEBUG
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# device = torch.device('cpu')

model_path = './models/gmlp_model.pth'
smpl_path = './data/samples.csv'

batch_size = 1
learning_rate = 1e-4
weight_decay = 1e-3
epoch_num = 100
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


def batched_train(train_loader, model):
    model.train()
    epoch_loss = 0.
    true_sample = 0
    # for i, (x, y, adj, batch, ptr) in enumerate(train_loader):
    #     x, y, adj, batch, ptr = x[1].reshape(
    #         (y[1].shape[0], -1, x[1].shape[1])), y[1], adj[1].reshape(
    #             (y[1].shape[0], -1, adj[1].shape[1])), batch[1], ptr[1]
    # for i, (x, y, adj, batch, ptr) in enumerate(train_loader):
    for i, data in enumerate(train_loader):
        # x, adj, y = data.x, data.adj, data.y
        x, edge_index, y = data.x, data.edge_index, data.y
        dense_y = torch.sparse_coo_tensor(edge_index,
                                          torch.ones(edge_index.shape[1]),
                                          torch.Size([x.shape[0], x.shape[0]]))
        adj = dense_y.to_dense()
        if adj.dim() == 2:
            x = x.reshape((1, x.shape[0], x.shape[1]))
            adj = adj.reshape((1, adj.shape[0], adj.shape[1]))
        # adj = adj @ adj
        x, adj, y = x.to(device), adj.to(device), y.to(device)
        out = model(x, adj, None)
        out = out.reshape((out.shape[0], -1))
        loss = model.loss(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        iter_true_sample = (out.argmax(dim=1).long() == y.long()). \
                float().sum().item()
        iter_acc = float(iter_true_sample) / out.shape[0]
        true_sample += iter_true_sample
    return true_sample, epoch_loss / len(train_loader)


def batched_test(test_loader, model):
    model.eval()
    true_sample = 0

    for i, data in enumerate(test_loader):
        x, edge_index, y = data.x, data.edge_index, data.y
        dense_y = torch.sparse_coo_tensor(edge_index,
                                          torch.ones(edge_index.shape[1]),
                                          torch.Size([x.shape[0], x.shape[0]]))
        adj = dense_y.to_dense()
        if adj.dim() == 2:
            x = x.reshape((1, x.shape[0], x.shape[1]))
            adj = adj.reshape((1, adj.shape[0], adj.shape[1]))
        x, adj, edge_index, y = x.to(device), adj.to(device), edge_index.to(
            device), y.to(device)
        out = model(x, adj, None)
        out = out.reshape((out.shape[0], -1))
        iter_true_sample = (out.argmax(dim=1).long() == y.long()). \
                float().sum().item()
        true_sample += iter_true_sample
    return true_sample


if __name__ == '__main__':
    # data_list = get_dataset(smpl_path, muti_target=False)
    data_list = get_benchmark_dataset()
    # random.shuffle(data_list)
    # train_list = data_list[:ceil(2 / 3 * len(data_list))]
    # test_list = data_list[ceil(2 / 3 * len(data_list)):]
    train_list, test_list, validation_list = random_split(
        data_list,
        (int(train_radio * len(data_list)), int(valid_radio * len(data_list)),
         (len(data_list) - int(train_radio * len(data_list)) -
          int(valid_radio * len(data_list)))))
    train_loader = DataLoader(train_list,
                              batch_size,
                              shuffle=True,
                              collate_fn=CollateFn(device))
    validation_loader = DataLoader(validation_list,
                                   batch_size,
                                   shuffle=True,
                                   collate_fn=CollateFn(device))
    test_loader = DataLoader(test_list,
                             batch_size,
                             shuffle=True,
                             collate_fn=CollateFn(device))
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
        true_sample, train_loss = batched_train(train_loader, model)
        train_acc = true_sample / len(train_list)
        validation_true_sample = batched_test(validation_loader, model)
        validation_acc = validation_true_sample / len(validation_list)
        tqdm.write(
            f"Epoch:{e+1}  \t train_acc:{train_acc:.2f}\t validation_acc:{validation_acc:.2f} \t train_loss:{train_loss:.2f}"
        )
        # print(
        #     # f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}'
        #     f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, loss: {loss.item():.4f}'
        # )
    # model.load_state_dict(torch.load(model_path))
    test_true_sample = batched_test(test_loader, model)
    test_acc = test_true_sample / len(test_list)
    print('test_acc', test_acc)
    # save last model
    if is_save_model:
        torch.save(model.state_dict(), model_path)
    #
    # train_model(model, data_loader)

# # model = GCN(hidden_channels=64)
# model = DiffPool(num_features=1, hidden_channels=64, num_classes=2).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()
# criterion = F.nll_loss()
# output, data.y.view(-1)