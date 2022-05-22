from random import sample
from unittest import result
import numpy as np
import pandas as pd
import os

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.wrappers import scikit_learn

from sklearn.ensemble import RandomForestClassifier
from sklearn import *

import threading
from multiprocessing.dummy import Pool as ThreadPool
import time


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
    def __init__(self, feature, p):
        self.feature_name = feature
        self.imp = np.array([p] * len(feature))


def compute_feature_importance(clf, feature):
    # init fi
    fi = feature_importance(feature, 0.0)
    tmp = clf.feature_importances_
    fi.imp[range(len(feature))] = tmp
    # indices of descending fi
    tmp = sorted(range(len(tmp)), key=lambda s: tmp[s], reverse=True)

    fi.feature_name = [feature[s] for s in tmp if fi.imp[s] != 0]
    return fi


def process(para_dict: dict):
    i = para_dict["i"]
    print(i)
    data_disease = para_dict["data_disease"]
    subjectID = para_dict["subjectID"]
    clf = para_dict["clf"]
    feature_importance = para_dict["fi"]
    data_raw = para_dict["data_raw"]
    #parameter-----------------------------------------------------------------------
    step = 0.05  #步长
    threshold_pro = 0.7  #第一轮最大健康概率，超过就停止
    threshold_num = 50  #最大微生物数量，超过就停止
    #parameter-----------------------------------------------------------------------

    patient = data_disease.iloc[[i]]
    id = subjectID.iloc[i]
    patient = patient.reset_index(drop=True)
    proba = clf.predict_proba(patient)
    max = proba[0, 0]
    t1 = np.array([a * step for a in range(int(1 / step) + 1)])
    # print("current abundance t1", t1)
    temp = patient.append([patient] * int(1 / step))
    # print("patient", patient)
    # print("temp", temp)
    features = []
    stable = False
    num = 0
    round = 0  #迭代轮数
    result = []

    while not stable:
        stable = True
        if round == 0:
            features = feature_importance.feature_name

        for f in features:
            previous = temp.iloc[0][f]
            temp[f] = t1
            # print("f", f)
            # print("temp[f]", temp[f])
            # print("clf.predict_proba(temp)", clf.predict_proba(temp))
            # print("clf.predict_proba(temp)[:, 0]",
            #   clf.predict_proba(temp)[:, 0])
            p = clf.predict_proba(temp)[:, 0]
            # print("p", p)
            # 一次尝试 1/step组，只要该组中有一个丰度能够使健康概率增加就选用该值
            if np.max(p) > max:
                max_pro = np.max(p)
                # 找到该值的序号
                max_num = np.argmax(p) * step
                # print("max_num", max_num)
                # 调整1/step组中该菌的丰度
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
                features = feature_importance.feature_name[:num]
                round = round + 1

                if stable:
                    treat = temp.iloc[0]
                    # 调整会原址
                    patient = patient * (data_raw.max() -
                                         data_raw.min()) + data_raw.min()
                    treat = treat * (data_raw.max() -
                                     data_raw.min()) + data_raw.min()

                    str = id + ":\n"
                    for feat in features:
                        if treat[feat] != patient[feat].values[0]:
                            str += "%s+%f -> %f\n" % (
                                feat, patient[feat].values[0], treat[feat])
                    result.append(str)
                    # print("I'm str", str)
                    #lock.acquire()
                    # np.record.write(str + "\n")
                    #lock.release()

                break

    return result


def disease_diagnosize(disease_name: str):

    file = "./back end/diagnosis/" + disease_name + ".csv"
    disease = "1:disease:" + "positive"

    data_raw = pd.read_csv(file, header=None, index_col=0, low_memory=False).T

    #删除所有特征中没有辨识度的特征（即某一特征在所有样本中都表现为同一值）
    feature = [s for s in data_raw.columns if "k__" in s]
    data_raw.drop(
        data_raw.loc[:, feature].columns[data_raw.loc[:, feature].max().astype(
            'float') == data_raw.loc[:, feature].min().astype('float')],
        axis=1,
        inplace=True)

    feature = [s for s in data_raw.columns if "k__" in s]
    if 'unclassified' in data_raw.columns:
        feature.append('unclassified')
    # 用于生成标签
    d = pd.DataFrame([disease.split(':')])
    label = pd.DataFrame([0] * len(data_raw))
    for i in range(len(d)):
        label[(data_raw[d.iloc[i,
                               1]].isin(d.iloc[i,
                                               2:])).tolist()] = int(d.iloc[i,
                                                                            0])
    # print(label.values.flatten())

    # 划分健康，患病病人
    temp = np.array(data_raw[d.iloc[0, 1]].isin(d.iloc[0, 2:]))
    data_healthy = data_raw.loc[~temp]
    data_disease = data_raw.loc[temp]
    subjectID = data_raw['subjectID'].loc[temp]
    # 数据预处理，只保留不是全零的物种，并且把值替换到0~1之间
    data_raw = data_raw.loc[:, feature].astype('float')
    data = (data_raw - data_raw.min()) / (data_raw.max() - data_raw.min())
    # 只取患者数据，同样做数据预处理
    data_disease = data_disease.loc[:, feature].astype('float')
    data_disease = (data_disease - data_raw.min()) / (data_raw.max() -
                                                      data_raw.min())

    clf = RandomForestClassifier(n_estimators=500,
                                 max_depth=None,
                                 min_samples_split=2,
                                 n_jobs=-1)
    clf.fit(data, label.values.flatten())
    feature_importance = compute_feature_importance(clf, feature)  #微生物重要度
    #global lock
    #num = len(data_disease)
    #lock = threading.Lock()
    para_list = []
    for i in range(len(data_disease)):
        para_list.append({
            "i": i,
            "data_disease": data_disease,
            "subjectID": subjectID,
            "clf": clf,
            "fi": feature_importance,
            "data_raw": data_raw
        })
    #begin_time = time()
    start = time.time()
    result = []
    pool = ThreadPool()
    print(f"total {len(data_disease)} patient")
    result.append(pool.map(process, para_list))

    pool.close()
    pool.join()
    end = time.time()
    result = result[0]
    # use this
    record = result[0][0].split(":")
    sample_i = record[0]
    treatment_i = record[1]
    # print('sample_i', sample_i, 'treatment_i', treatment_i)
    # print("type(treatment_i)", type(treatment_i))
    # print("spilted", str(treatment_i).split("\n"))
    # print("I am result len", len(result))
    return result, "共运行了 " + str(end - start)[:8] + " s"


if __name__ == "__main__":
    disease_name = "ibd"
    _, here = disease_diagnosize(disease_name)
    print(here)