from typing import Dict
import torch
import time
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

smpl_path = 'cirrhosis'

only_evaluation = False


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


def batched_train(loader, model, optimizer):
    model.train()
    epoch_loss = 0.
    iter_acc = iter_auc = 0.
    timer = 0
    for _, batch in enumerate(loader):
        # 这里解决了一个问题，之前是放在batch之外的
        loss = 0.
        lables_list = []
        probs_list = []
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
            iter_acc = iter_acc + metrics.accuracy_score(
                y.cpu().detach(),
                out.cpu().detach().argmax(dim=1))
            lables_list.append(y.cpu().detach()[0])
            probs_list.append(out.cpu().detach()[0][1])
            # iter_acc = iter_acc + metrics.precision_score(
            #     y.cpu().detach(), y.cpu().detach(), average='micro')
            # if y.cpu().detach() == 1 and out.cpu().detach().argmax(dim=1) == 1:
            #     iter_auc += 1
        loss.backward()
        lables = np.array(lables_list)
        probs = np.array(probs_list)
        try:
            iter_auc += metrics.roc_auc_score(lables, probs)
        except:
            iter_auc += 0.5
        # 梯度截断防止梯度爆炸导致过拟合，类似于dropout
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    return epoch_loss / len(loader), iter_acc / timer, iter_auc / len(loader)


def calcAUC_byProb(labels, probs):
    N = 0  # 正样本数量
    P = 0  # 负样本数量
    neg_prob = []  # 负样本的预测值
    pos_prob = []  # 正样本的预测值
    for index, label in enumerate(labels):
        if label == 1:
            # 正样本数++
            P += 1
            # 把其对应的预测值加到“正样本预测值”列表中
            pos_prob.append(probs[index][0][1])
            print("positive", probs[index][0][1])
        else:
            # 负样本数++
            N += 1
            # 把其对应的预测值加到“负样本预测值”列表中
            neg_prob.append(probs[index][0][0])
            print("negative", probs[index][0][0])
    number = 0.
    # 遍历正负样本间的两两组合
    for pos in pos_prob:
        for neg in neg_prob:
            # 如果正样本预测值>负样本预测值，正序对数+1
            print("pos", pos, "neg", neg)
            if (pos > neg):
                number += 1
            # 如果正样本预测值==负样本预测值，算0.5个正序对
            elif (pos == neg):
                number += 0.5
    return number / (N * P)


# @torch.no_grad
def batched_test(loader, model, testing: bool = False):
    model.eval()
    iter_acc = iter_auc = 0.
    timer = 0
    for _, batch in enumerate(loader):
        lables_list = []
        probs_list = []
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

            lables_list.append(y.cpu().detach()[0])
            probs_list.append(out.cpu().detach()[0][1])
            iter_acc = iter_acc + metrics.accuracy_score(
                y.cpu().detach(),
                out.cpu().detach().argmax(dim=1))
        lables = np.array(lables_list)
        probs = np.array(probs_list)
        try:
            iter_auc += metrics.roc_auc_score(lables, probs)
        except:
            iter_auc += 0.5

    return iter_acc / timer, iter_auc / len(loader)


def graph_model_evaluation(data_dict: dict):

    batch_size = int(data_dict["batchsize"])
    epoch_num = int(data_dict["epoch"])
    threshold = int(data_dict["threshold"])
    feature = int(data_dict["feature"])
    disease = data_dict["dataset"]
    only_evaluation = bool(data_dict["only_evaluation"])

    is_loading_model = True

    learning_rate = 1e-3
    train_radio = 0.6
    valid_radio = 0.5 - train_radio / 2
    model_path = './models/' + disease + ".pth"

    data_list = get_dataset(disease, threshold, feature)
    # data_list = get_benchmark_dataset('PROTEINS')
    train_list, test_list, validation_list = random_split(
        data_list, [
            int(train_radio * len(data_list)),
            int(valid_radio * len(data_list)),
            len(data_list) - int(train_radio * len(data_list)) -
            int(valid_radio * len(data_list))
        ],
        generator=torch.Generator().manual_seed(520))

    train_loader = DataLoader(train_list, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_list, batch_size)
    test_loader = DataLoader(test_list, batch_size)

    in_feature = train_list[0].x.shape[1]
    classes = max([item.y for item in data_list]) + 1
    maxmum_nodes = max([item.x.shape[0] for item in data_list])
    pool_size = ceil(maxmum_nodes * 0.25)

    model = BatchedModel(pool_size,
                         input_shape=in_feature,
                         n_classes=int(classes),
                         device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0.
    toc1 = time.perf_counter()

    if ~only_evaluation:
        for e in tqdm(range(epoch_num)):
            train_loss, train_acc, train_auc = batched_train(
                train_loader, model, optimizer)
            valid_acc, valid_auc = batched_test(validation_loader, model)
            if valid_acc >= best_acc:
                best_acc = valid_acc
                # state = {
                #     'net': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'epoch': e
                # }
                torch.save(model, model_path)
            tqdm.write(
                f'''Epoch:{e+1} \t train: {train_acc:.3f}\t {train_auc:.3f} \t valid: {valid_acc:.3f}\t {valid_auc:.3f} \t {train_loss:.3f}'''
            )
    toc2 = time.perf_counter()

    if is_loading_model:
        model = torch.load(model_path)
        print('load done')
    test_acc, test_auc = batched_test(test_loader, model, optimizer, True)
    toc3 = time.perf_counter()

    print(
        f'''test_acc, {test_acc:.3f} {test_auc:.3f}, max_valid_acc, {best_acc:.3f},runtime,{toc3-toc2}'''
    )
    # save last model
    # if is_save_model:

    # load best model


if __name__ == "__main__":
    data_dict = {
        "batchsize": "16",
        "epoch": "150",
        "threshold": "50",
        "feature": "90",
        "dataset": "cirrhosis",
        "only_evaluation": "False",
    }
    graph_model_evaluation(data_dict)
