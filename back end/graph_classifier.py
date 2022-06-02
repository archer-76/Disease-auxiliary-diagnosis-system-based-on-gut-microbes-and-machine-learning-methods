from cProfile import label
import torch
import time
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
# import torch_geometric.nn.DataParallel as dp
from math import ceil
import torch.optim as optim
from tqdm import tqdm
from utils import *
from models import *
from sklearn import metrics
import sqlite3
from sklearn.model_selection import StratifiedKFold
# DEBUG
import os
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
    lables_list = []
    probs_list = []
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

        # 梯度截断防止梯度爆炸导致过拟合，类似于dropout
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    lables = np.array(lables_list)
    probs = np.array(probs_list)
    try:
        iter_auc += metrics.roc_auc_score(lables, probs)
    except:
        iter_auc += 0.5
        print("TRAIN auc error")

    return epoch_loss / len(loader), iter_acc / timer, iter_auc


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
    lables_list = []
    probs_list = []
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
        print("test auc error")

    return iter_acc / timer, iter_auc


def graph_model_evaluation(data_dict: dict, diagnosizing=False):

    print(torch.cuda.is_available())

    batch_size = int(data_dict["batchsize"])
    epoch_num = int(data_dict["epoch"])
    threshold = int(data_dict["threshold"])
    feature = int(data_dict["feature"])
    disease = data_dict["dataset"]
    diagnosizing = diagnosizing

    disease_path = "./back end/data/" + disease + ".csv"
    diagnosis_path = "./back end/diagnosis/" + disease + ".csv"

    is_loading_model = True
    accs, aucs = [], []
    result_list = []
    best_acc, average_acc, average_auc = 0., 0., 0.
    learning_rate = 1e-3
    train_radio = 0.6
    valid_radio = 0.5 - train_radio / 2
    model_path = './models/' + disease + ".pth"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EvaluationHistory_path = os.path.join(BASE_DIR, "EvaluationHistory.db")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_list = get_dataset(disease_path, disease, threshold, feature)
    if diagnosizing:
        diagnosis_list = get_dataset(diagnosis_path, disease, threshold,
                                     feature)

    in_feature = data_list[0].x.shape[1]
    classes = max([item.y for item in data_list]) + 1
    maxmum_nodes = max([item.x.shape[0] for item in data_list])
    pool_size = ceil(maxmum_nodes * 0.25)

    conn = sqlite3.connect(EvaluationHistory_path)
    c = conn.cursor()
    cursor = c.execute(
        '''
            SELECT MAX(acc)
            FROM EvaluationHistory
            WHERE classifier = ? AND dataset = ?         
            ''', ["GNN", disease])
    cursors = cursor.fetchall()
    for record in cursors:
        tmp_list = list(record)
        if tmp_list != [None]:
            best_acc = float(tmp_list[0])
    print("best acc = ", best_acc)

    label = []
    for data in data_list:
        label.append(int(data.y[0]))

    skf = StratifiedKFold(5, shuffle=True, random_state=1)
    skf_split = skf.split(np.zeros(len(label)), label)
    # data_list = get_benchmark_dataset('PROTEINS')
    # train_list, test_list, validation_list = random_split(
    #     data_list, [
    #         int(train_radio * len(data_list)),
    #         int(valid_radio * len(data_list)),
    #         len(data_list) - int(train_radio * len(data_list)) -
    #         int(valid_radio * len(data_list))
    #     ],
    #     generator=torch.Generator().manual_seed(520))
    # train_loader = DataLoader(train_list, batch_size, shuffle=True)
    # validation_loader = DataLoader(validation_list, batch_size)
    # test_loader = DataLoader(test_list, batch_size)
    if diagnosizing:
        diagnosis_loader = DataLoader(diagnosis_list, batch_size)
    if (not diagnosizing):
        for train_index, test_index in skf_split:
            test_list, train_list = [], []
            for s in train_index:
                train_list.append(data_list[s])
            for s in test_index:
                test_list.append(data_list[s])

            train_loader = DataLoader(train_list, batch_size)
            test_loader = DataLoader(test_list, batch_size)

            model = BatchedModel(pool_size,
                                 input_shape=in_feature,
                                 n_classes=int(classes),
                                 device=device).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            for e in tqdm(range(epoch_num)):
                train_loss, train_acc, train_auc = batched_train(
                    train_loader, model, optimizer)
                tqdm.write(
                    f'''Epoch:{e+1} \tacc: {train_acc:.3f}\tauc:{train_auc:.3f} \tloss: {train_loss:.3f}'''
                )
            valid_acc, valid_auc = batched_test(test_loader, model)
            print(f"valid\tacc: {valid_acc:.3f}\tauc: {valid_auc:.3f}")
            accs.append(valid_acc)
            aucs.append(valid_auc)
            if valid_acc >= best_acc:
                print("当前模型更加优秀！", valid_acc, "超过了", best_acc)
                best_acc = valid_acc
                torch.save(model, model_path)
        print("average acc: %.3f (+/- %.3f)" % (np.mean(accs), np.std(accs)))
        print("average auc: %.3f (+/- %.3f)" % (np.mean(aucs), np.std(aucs)))
        average_acc = np.mean(accs)
        average_auc = np.mean(aucs)
    toc2 = time.perf_counter()
    if is_loading_model:
        print("torch.cuda.is_available()", torch.cuda.is_available())
        model = torch.load(model_path)
        print('load done')
    if diagnosizing:
        model.eval()
        diagnosis_df = pd.read_csv(diagnosis_path,
                                   header=None,
                                   index_col=0,
                                   low_memory=False).T
        new_pred = []
        for _, batch in enumerate(diagnosis_loader):
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
                pred = out.cpu().detach().argmax(dim=1).reshape(-1)
                new_pred.append(int(pred[0]))
        print(new_pred)
        diagnosis_df['disease'] = new_pred
        diagnosis_df['disease'].replace(1, 'positive', inplace=True)
        diagnosis_df['disease'].replace(0, 'negative', inplace=True)

        for i in range(len(diagnosis_df)):
            dd = diagnosis_df.iloc[i]
            result_list.append((dd['subjectID'], dd['gender'], dd['age'],
                                dd['country'], dd['disease']))
        diagnosis_df.T.to_csv(diagnosis_path, header=None)

    # test_acc, test_auc = batched_test(test_loader, model, True)
    toc3 = time.perf_counter()

    # print(
    #     f'''test_acc, {test_acc:.3f} {test_auc:.3f}, max_valid_acc, {best_acc:.3f},runtime,{toc3-toc2}'''
    # )

    return str(average_acc)[:6], str(average_auc)[:6], result_list
    # save last model
    # if is_save_model:

    # load best model


if __name__ == "__main__":
    data_dict = {
        "batchsize": "16",
        "epoch": "1",
        "threshold": "50",
        "feature": "90",
        "dataset": "cirrhosis",
    }
    _, _, result_list = graph_model_evaluation(data_dict, False)
