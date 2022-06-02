from unittest import result
import numpy as np
import pandas as pd
import os
from keras.layers import *

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.wrappers import scikit_learn
import tensorflow.keras.backend as backend

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class feature_importance:
    def __init__(self, feature, p):
        self.feat_name = feature
        self.imp = np.array([p] * len(feature))


def compute_feature_importance(clf, feature):
    # init fi
    fi = feature_importance(feature, 0.0)
    tmp = clf.feature_importances_
    fi.imp[range(len(feature))] = tmp
    # indices of descending fi
    tmp = sorted(range(len(tmp)), key=lambda s: tmp[s], reverse=True)

    fi.feat_name = [feature[s] for s in tmp if fi.imp[s] != 0]
    return fi


def dnn_model(input_d, c_num):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=input_d))
    #model.add(Dropout(0.3))
    #model.add(Dense(256,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(c_num, activation='softmax'))

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model


def cnn_model(input_d, c_num):
    model = Sequential()
    model.add(
        Conv1D(64,
               3,
               padding='valid',
               activation='relu',
               input_shape=(input_d, 1)))
    #model.add(MaxPooling1D(pool_size=2,padding='valid'))
    #model.add(Conv1D(32,3,padding='valid',activation='relu'))
    #model.add(MaxPooling1D(pool_size=2,padding='valid'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='feature'))
    #model.add(Dropout(0.5))
    model.add(Dense(c_num, activation='softmax'))

    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


def basic_model_inferrence(para_dict: dict, diagnosizing: bool = False):

    diagnosis_path = "./back end/diagnosis/" + para_dict["dataset"] + ".csv"
    if para_dict != {}:
        evaluation_path = "./back end/data/" + para_dict["dataset"] + ".csv"
        disease = "1:disease:" + para_dict["dataset"]
        classifier = para_dict["classifier"]
        epoch = int(para_dict["epoch"])
        batchsize = int(para_dict["batchsize"])
        feature = float(para_dict["feature"]) / 100
        print(
            "paralist",
            [evaluation_path, disease, classifier, epoch, batchsize, feature])
    else:
        evaluation_path = "./back end/data/t2d.csv"
        disease = "1:disease:t2d"
        classifier = "1D-CNN"
        epoch = 4
        batchsize = 16
        feature = 0  #取前feature个特征进行训练,为0则不选择
        print(
            "paralist",
            [evaluation_path, disease, classifier, epoch, batchsize, feature])
    runs_round = 1

    data = pd.DataFrame([])
    diagnosis_num = evaluation_num = 0

    evaluation_df = pd.read_csv(evaluation_path,
                                header=None,
                                index_col=0,
                                low_memory=False).T
    evaluation_num += evaluation_df.shape[0]
    diagnosis_df = pd.read_csv(diagnosis_path,
                               header=None,
                               index_col=0,
                               low_memory=False).T
    diagnosis_num += diagnosis_df.shape[0]

    data = data.append(evaluation_df)
    data = data.append(diagnosis_df)

    #删除所有特征中没有辨识度的特征（即某一特征在所有样本中都表现为同一值）
    feat = [s for s in data.columns if "k__" in s]
    data.drop(data.loc[:, feat].columns[data.loc[:, feat].max().astype(
        'float') == data.loc[:, feat].min().astype('float')],
              axis=1,
              inplace=True)

    feat = [s for s in data.columns if "k__" in s]
    if 'unclassified' in data.columns:
        feat.append('unclassified')
    k = int(feature * len(feat))

    #样本标签
    df = pd.DataFrame([disease.split(':')])
    label = pd.DataFrame([0] * len(data))
    label[(data[df.iloc[0, 1]].isin(df.iloc[0, 2:])).tolist()] = 1
    #print(label.values.flatten())

    class_num = np.max(label.values) + 1  #类别数
    # 讲特征的更新同步到数据
    data = data.loc[:, feat].astype('float')
    data = (data - data.min()) / (data.max() - data.min())

    evaluation_data = data.iloc[:evaluation_df.shape[0]]
    diagnosis_data = data.iloc[evaluation_df.shape[0]:]
    evaluation_label = label.iloc[:evaluation_df.shape[0]]

    accs = []
    aucs = []

    if feature != 0:
        clf = RandomForestClassifier(n_estimators=500,
                                     max_depth=None,
                                     min_samples_split=2,
                                     n_jobs=-1)
        clf.fit(evaluation_data, evaluation_label.values.flatten())
        fi = compute_feature_importance(clf, feat)
        evaluation_data = evaluation_data[fi.feat_name[:k]]
        diagnosis_data = diagnosis_data[fi.feat_name[:k]]

    for i in range(runs_round):
        skf = StratifiedKFold(10, shuffle=True, random_state=i)
        skf_split = skf.split(np.zeros(label[:evaluation_num].shape),
                              label[:evaluation_num])

        for train_index, test_index in skf_split:
            backend.clear_session()  #清除上一次会话

            i_tr = np.array([False] * evaluation_num)
            for s in train_index:
                i_tr[s] = True

            train_data = evaluation_data.loc[i_tr]
            train_label = evaluation_label.loc[i_tr]

            val_data = evaluation_data.loc[~i_tr]
            val_label = evaluation_label.loc[~i_tr]

            if classifier == "DNN":
                clf = scikit_learn.KerasClassifier(build_fn=dnn_model,
                                                   epochs=epoch,
                                                   verbose=2,
                                                   batch_size=batchsize,
                                                   input_d=train_data.shape[1],
                                                   c_num=class_num)
                clf.fit(train_data,
                        train_label,
                        validation_data=(val_data, val_label))

            elif classifier == "1D-CNN":
                train_data = np.expand_dims(train_data, axis=2)
                val_data = np.expand_dims(val_data, axis=2)
                print("val_data.shape", val_data.shape)

                clf = scikit_learn.KerasClassifier(build_fn=cnn_model,
                                                   epochs=epoch,
                                                   verbose=2,
                                                   batch_size=batchsize,
                                                   input_d=train_data.shape[1],
                                                   c_num=class_num)
                clf.fit(train_data,
                        train_label,
                        validation_data=(val_data, val_label))

            elif classifier == "RF":
                clf = RandomForestClassifier(n_estimators=500,
                                             max_depth=None,
                                             min_samples_split=2,
                                             n_jobs=-1)

                clf.fit(train_data, train_label.values.flatten())  #rf

            elif classifier == "SVM":
                clf = SVC(C=2, kernel='rbf', probability=True, gamma=2**-3)
                # print("train_data", train_data.shape, "train_label",
                #       train_label.shape)
                clf.fit(train_data, train_label.values.flatten())  #rf

            val_pred = clf.predict(val_data)
            acc = metrics.accuracy_score(val_label, val_pred)
            accs.append(acc)
            print("acc= %.3f on validation set" % acc)

            #f1 = metrics.f1_score(val_label, val_pred, pos_label=None, average='weighted')
            #f1_scores.append(f1)

            #recall = metrics.recall_score(val_label, val_pred, pos_label=None, average='weighted')
            #recalls.append(recall)

            #precision = metrics.precision_score(val_label, val_pred, pos_label=None, average='weighted')
            #precisions.append(precision)

            if class_num == 2:
                val_scores = clf.predict_proba(val_data)[:, 1]
                # print("shapes", val_label.shape, val_scores.shape,
                #   clf.predict_proba(val_data).shape)
                auc = metrics.roc_auc_score(val_label, val_scores)
                aucs.append(auc)
                # print("auc= %.3f on validation set" % auc)

    result_list = []
    if diagnosizing:
        print("diagnosis_data.shape", diagnosis_data.shape)
        if classifier == "1D-CNN":
            diagnosis_data = np.expand_dims(diagnosis_data, axis=2)
        #pickle model to disk
        # rfclf = RandomForestClassifier(n_estimators=500,
        #                                max_depth=None,
        #                                min_samples_split=2,
        #                                n_jobs=-1)
        # rfclf.fit(evaluation_data, evaluation_label.values.flatten())  #rf
        # joblib.dump(rfclf, 'my_randomforest_model.joblib')
        new_pred = clf.predict(diagnosis_data)
        print("new_pred", new_pred)
        diagnosis_df['disease'] = new_pred
        diagnosis_df['disease'].replace(1, 'positive', inplace=True)
        diagnosis_df['disease'].replace(0, 'negative', inplace=True)

        for i in range(len(diagnosis_df)):
            dd = diagnosis_df.iloc[i]
            result_list.append((dd['subjectID'], dd['gender'], dd['age'],
                                dd['country'], dd['disease']))
        diagnosis_df.T.to_csv(diagnosis_path, header=None)
    # print(result_list)
    print("average acc: %.3f (+/- %.3f)" % (np.mean(accs), np.std(accs)))
    average_acc = np.mean(accs)
    #print("average precision: %.4f (+/- %.4f)" % (np.mean(precisions), np.std(precisions)))
    #print("average recall: %.4f (+/- %.4f)" % (np.mean(recalls), np.std(recalls)))
    #print("average f1: %.4f (+/- %.4f)" % (np.mean(f1_scores), np.std(f1_scores)))

    average_auc = np.mean(aucs)
    print("average auc: %.3f (+/- %.3f)" % (np.mean(aucs), np.std(aucs)))

    average_acc = str(average_acc)[:6]
    average_auc = str(average_auc)[:6]
    return average_acc, average_auc, result_list


# 1D-CNN DNN
if __name__ == "__main__":
    para_dict = {
        "dataset": "wt2d",
        "classifier": "SVM",
        "epoch": "5",
        "batchsize": "32",
        "feature": "90",
    }
    # para_dict = {}
    basic_model_inferrence(para_dict, True)
