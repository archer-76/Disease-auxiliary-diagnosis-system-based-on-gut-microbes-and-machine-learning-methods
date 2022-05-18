path = "./back end/DiagnosisHistory"
import os
import random
import numpy as np
from sklearn.metrics import auc
from yaml import load

# Save
dicta = {'a': 1, 'b': 2, 'c': 3}


def model_diagnosize(data_dict: dict, sample_path: str):
    para_list = [item for item in data_dict.values()]
    dataset = data_dict["dataset_value"]
    classifier = data_dict["classifier_value"]
    epoch = int(data_dict['epoch'])
    batchsize = int(data_dict['batchsize'])
    # could not convert string to float: 'None',前端返回结果就是这样
    if (data_dict["threshold"] != ""):
        threshold = float(data_dict["threshold"])
        graph_classifier = data_dict["graph_classifier_value"]
    else:
        threshold = graph_classifier = "_"
    record = ""
    acc = str(random.uniform(0.5, 1))
    auc = str(random.uniform(0.5, 1))
    acc = acc[:4]
    auc = auc[:4]
    for item in data_dict.values():
        if type(item) != str:
            print("found one")
            item = str(item)
        record += item
        record += "-"
    diagnosis_result = [
        ['男', "32", "天国", "positive"],
        ['女', "23", "内国", "positive"],
        ['男', "23", "外国", "negative"],
    ]
    specfic_result = ['女', "32", "地国", "negative"]
    diagnosis_result.append(specfic_result)
    return diagnosis_result, acc, auc


def savedict_tofile(path: str, record: str, dict: dict):
    path = path + "/" + record
    if not os.path.exists(path):
        os.mkdir(path)
    # os.mkdir("/hourly")
    child_path = path + "/data.npy"
    np.save(child_path, dict)  # 注意带上后缀名


def loaddict_fromfile(path: str, file_name: str):
    child_path = path + file_name
    load_dict = np.load(child_path, allow_pickle=True).item()
    print(load_dict)


savedict_tofile(path, "Colorectal-RF-30-_-100-16-None-0.9-0.9-", dicta)
