import numpy as np
import pandas as pd
import os
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.wrappers import scikit_learn
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import *

import threading
from multiprocessing.dummy import Pool as ThreadPool


def cnn_model(input_d, c_num):
    model = Sequential()
    model.add(
        Conv1D(64,
               3,
               padding='valid',
               activation='relu',
               input_shape=(input_d, 1)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', name='feature'))
    model.add(Dense(c_num, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model


class feature_importance:
    def __init__(self, feat, p):
        self.feat_sel = feat
        self.imp = np.array([p] * len(feat))


def compute_feature_importance(el, feat):
    fi = feature_importance(feat, 0.0)
    t = el.feature_importances_
    fi.imp[range(len(feat))] = t

    t = sorted(range(len(t)), key=lambda s: t[s], reverse=True)
    fi.feat_sel = [feat[s] for s in t if fi.imp[s] != 0]

    return fi


def process(task):
    print(task)
    #parameter-----------------------------------------------------------------------
    step = 0.05  #步长
    threshold_pro = 0.7  #第一轮最大健康概率，超过就停止
    threshold_num = 50  #最大微生物数量，超过就停止
    #parameter-----------------------------------------------------------------------

    patient = data_disease.iloc[[i]]
    id = subjectID.iloc[i]
    patient = patient.reset_index(drop=True)

    max = clf.predict_proba(patient)[0, 0]
    t1 = np.array([a * step for a in range(int(1 / step) + 1)])
    temp = patient.append([patient] * int(1 / step))

    features = []
    stable = False
    num = 0
    round = 0  #迭代轮数

    while not stable:
        stable = True
        if round == 0:
            features = fi.feat_sel

        for f in features:
            previous = temp.iloc[0][f]
            temp[f] = t1
            p = clf.predict_proba(temp)[:, 0]

            if np.max(p) > max:
                max_pro = np.max(p)
                max_num = np.argmax(p) * step

                temp[f] = [max_num] * (int(1 / step) + 1)
                max = max_pro
                stable = False

            else:
                temp[f] = [previous] * (int(1 / step) + 1)

            if round == 0:
                num = num + 1

            #每次迭代结束时
            if (round == 0 and
                (max > threshold_pro or num > threshold_num)) or (
                    round != 0 and f is features[-1]):
                features = fi.feat_sel[:num]
                round = round + 1

                if stable:
                    treat = temp.iloc[0]
                    patient = patient * (data_raw.max() -
                                         data_raw.min()) + data_raw.min()
                    treat = treat * (data_raw.max() -
                                     data_raw.min()) + data_raw.min()

                    str = "sample " + id + ":\n"
                    for feat in features:
                        if treat[feat] != patient[feat].values[0]:
                            str += "%s\n %f -> %f\n" % (
                                feat, patient[feat].values[0], treat[feat])

                    #lock.acquire()
                    fidout.write(str + "\n")
                    #lock.release()

                break


if __name__ == "__main__":
    fidout = open('ibd_diagnosis.txt', 'w')
    file = "./back end/data/ibd.txt"
    disease = "1:disease:ibd"
    classifier = "rf"
    ep = 5
    bs = 16

    data_raw = pd.read_csv(file,
                           sep='\t',
                           header=None,
                           index_col=0,
                           low_memory=False).T

    #删除所有特征中没有辨识度的特征（即某一特征在所有样本中都表现为同一值）
    feat = [s for s in data_raw.columns if "k__" in s]
    data_raw.drop(
        data_raw.loc[:, feat].columns[data_raw.loc[:, feat].max().astype(
            'float') == data_raw.loc[:, feat].min().astype('float')],
        axis=1,
        inplace=True)

    feat = [s for s in data_raw.columns if "k__" in s]
    if 'unclassified' in data_raw.columns:
        feat.append('unclassified')
    # 用于生成标签
    d = pd.DataFrame([s.split(':') for s in disease.split(',')])
    label = pd.DataFrame([0] * len(data_raw))
    for i in range(len(d)):
        label[(data_raw[d.iloc[i,
                               1]].isin(d.iloc[i,
                                               2:])).tolist()] = int(d.iloc[i,
                                                                            0])
    # print(label.values.flatten())

    #divide healthy and sick sample
    temp = np.array(data_raw[d.iloc[0, 1]].isin(d.iloc[0, 2:]))
    data_healthy = data_raw.loc[~temp]
    data_disease = data_raw.loc[temp]
    subjectID = data_raw['subjectID'].loc[temp]
    # 数据预处理，只保留不是全零的物种，并且把值替换到0~1之间
    data_raw = data_raw.loc[:, feat].astype('float')
    data = (data_raw - data_raw.min()) / (data_raw.max() - data_raw.min())
    # 只取患者数据，同样做数据预处理
    data_disease = data_disease.loc[:, feat].astype('float')
    data_disease = (data_disease - data_raw.min()) / (data_raw.max() -
                                                      data_raw.min())

    clf = RandomForestClassifier(n_estimators=500,
                                 max_depth=None,
                                 min_samples_split=2,
                                 n_jobs=-1)
    clf.fit(data, label.values.flatten())
    fi = compute_feature_importance(clf, feat)  #微生物重要度
    #global lock
    #num = len(data_disease)
    #lock = threading.Lock()

    task_list = []
    for i in range(len(data_disease)):
        task_list.append({"i": i, "dataset": "data"})

    start = time.time()
    #MutiThread(4)
    pool = ThreadPool()
    pool.map(process, task_list)
    pool.close()
    pool.join()
    fidout.close()
    end = time.time()
    print(end - start)
