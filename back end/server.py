from random import sample
from unittest import result
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS
import json
from test_file import *
import sys
import sqlite3
from sqlite3.dbapi2 import Cursor
import os.path
import datetime
import pandas as pd
from disease_classifier import *
from disease_diagnosizer import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DiagnosisHistory_path = os.path.join(BASE_DIR, "DiagnosisHistory.db")
EvaluationHistory_path = os.path.join(BASE_DIR, "EvaluationHistory.db")
Sample_path = os.path.join(BASE_DIR, "./diagnosis/samples.csv")

app = Flask(__name__)
# cors = CORS(app, resources={r"getMsg": {"origins": "*"}})
CORS(app, supports_credentials=True)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/BestHistory', methods=['GET', 'POST'])
def BestHistory():
    record_list = []
    conn = sqlite3.connect(EvaluationHistory_path)
    clf_list = ['RF', 'SVM', 'GNN', 'DNN', '1D-CNN']
    for current_clf in clf_list:
        c = conn.cursor()
        cursor = c.execute(
            '''
            SELECT *
            FROM EvaluationHistory
            WHERE classifier = ? AND acc in (
                        SELECT MAX(acc)
                        FROM EvaluationHistory
                        WHERE classifier = ? 
                    ) 
                ''', [current_clf, current_clf])
        cursors = cursor.fetchall()
        for record in cursors:
            tmp_list = list(record)
            for i in range(len(record)):
                tmp_list[i] = str(tmp_list[i])
        record_list.append(tmp_list)
    # savedict_tofile(path, record, data_dict)
    conn.close()
    print('BestHistory', record_list)
    response = {'record': record_list}
    return jsonify(response)


@app.route('/TrainHistory', methods=['GET', 'POST'])
def TrainHistory():
    record_list = []
    conn = sqlite3.connect(EvaluationHistory_path)
    c = conn.cursor()
    cursor = c.execute('''
        SELECT *
        FROM EvaluationHistory
            ''')
    cursors = cursor.fetchall()
    for record in cursors:
        tmp_list = list(record)
        for i in range(len(record)):
            tmp_list[i] = str(tmp_list[i])
        record_list.append(tmp_list)
    print('TrainHistory', record_list)
    # savedict_tofile(path, record, data_dict)
    conn.close()
    response = {'record': record_list}
    return jsonify(response)


@app.route('/ModelEvaluation', methods=['GET', 'POST'])
def ModelEvaluation():
    if request.method == 'GET':  # 获取vue中传递的值

        response = {
            'Cirrhosis': ['RF', '30', 'None', '100', '16', 'None'],
            'IBD': ['RF', '30', 'None', '100', '16', 'None'],
            'Colorectal': ['RF', '30', 'None', '100', '16', 'None'],
            'Obesity': ['RF', '30', 'None', '100', '16', 'None'],
            'T2D': ['RF', '64', 'None', '100', '16', 'None'],
            'WT2D': ['GNN', '64', 'DiffPool', '100', '16', '36'],
        }
    elif request.method == 'POST':
        data = request.get_data()
        data_dict = json.loads(data)
        # 假定读取完为一个字典
        record_list = [str(item) for item in data_dict.values()]
        print("eva datadict", data_dict)
        current_time = datetime.datetime.now()
        str_time = current_time.strftime('%Y%m%d%H%M%S%f')[:-3]
        HistoryID = "{0}".format(str_time)
        record_list.insert(0, HistoryID)
        acc, auc, _ = basic_model_inferrence(data_dict, False)
        print('evaluation result', acc, auc)
        record_list.append(acc)
        record_list.append(auc)
        record_list.append(
            datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        print('Evaluation', record_list)
        response = {'finished': True, 'acc': acc, 'auc': auc}
        conn = sqlite3.connect(EvaluationHistory_path)
        c = conn.cursor()
        cursor = c.execute(
            '''
            insert into
            EvaluationHistory
            values(?,?,?,?,?,?,?,?,?,?,?)
                ''', record_list)
        cursors = cursor.fetchall()
        for record in cursors:
            pass
        # savedict_tofile(path, record, data_dict)
        conn.commit()
        conn.close()

    return jsonify(response)


@app.route('/HandleFileRequest', methods=['GET', 'POST'])
def HandleFileRequest():

    file0 = request.form.get(
        'fileToUpload'
    )  # request.form outputs ImmutableMultiDict([]); request.form.get('fileToUpload') outputs None
    file = request.files.getlist('fileToUpload')[
        0]  # the type of file is FileStorage
    upload_dataset = pd.read_csv(
        file,
        header=None,
    )
    filepath = "./back end/diagnosis/" + file.filename
    upload_dataset.to_csv(filepath, index=False, header=False)
    response = {
        'finished': True,
    }
    return jsonify(response)


@app.route('/DieaseDiagnosize', methods=['GET', 'POST'])
def DieaseDiagnosize():
    if request.method == 'GET':  # 获取vue中传递的值
        #从评估记录中找到acc最高的模型
        record_list = []

        dis_list = ['cirrhosis', 'ibd', 'colorectal', 'obesity', 't2d', 'wt2d']
        for current_dis in dis_list:
            conn = sqlite3.connect(EvaluationHistory_path)
            c = conn.cursor()
            cursor = c.execute(
                '''
                SELECT dataset,classifier,feature,pooling,epoch,batchsize,threshold
                FROM EvaluationHistory
                WHERE dataset = ? AND acc in (
                            SELECT MAX(acc)
                            FROM EvaluationHistory
                            WHERE dataset = ? 
                        ) 
                    ''', [current_dis, current_dis])
            cursors = cursor.fetchall()
            for record in cursors:
                tmp_list = list(record)
                for i in range(len(record)):
                    tmp_list[i] = str(tmp_list[i])
                record_list.append(tmp_list)
        # savedict_tofile(path, record, data_dict)
        conn.close()
        print('DieaseDiagnosize', record_list)
        response = {'record': record_list}
    elif request.method == 'POST':
        data = request.get_data()
        data_dict = json.loads(data)
        disease_name = data_dict["dataset"]
        # 假定读取完为一个字典
        record_list = [str(item) for item in data_dict.values()]
        current_time = datetime.datetime.now()
        str_time = current_time.strftime('%Y%m%d%H%M%S%f')[:-3]
        HistoryID = "{0}".format(str_time)
        record_list.insert(0, HistoryID)
        # 结果为一个二维列表，第二维度中装了specfichistory中的信息
        # 输入：OTU表路径和一堆参数，输出：二维列表(sample_num,feature_num)
        _, _, diagnosis_result = basic_model_inferrence(data_dict, True)
        record_list.append(
            datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        response = {
            'finished': True,
        }
        print(len(record_list), record_list)
        conn = sqlite3.connect(DiagnosisHistory_path)
        c = conn.cursor()
        cursor = c.execute(
            '''
            insert into
            DiagnosisHistory
            values(?,?,?,?,?,?,?,?,?)
                ''', record_list)
        cursors = cursor.fetchall()
        # savedict_tofile(path, record, data_dict)
        conn.commit()
        for i in range(len(diagnosis_result)):
            print(diagnosis_result[i])
            specfic_result = list(diagnosis_result[i])
            specfic_result.append(HistoryID)
            cursor = c.execute(
                '''
                insert into
                SpecficDiagnosis
                values(?,?,?,?,?,?)
                ''', specfic_result)
            conn.commit()

        # diagnosize
        treatment_result, _ = disease_diagnosize(disease_name)
        for i in range(len(treatment_result)):
            record = treatment_result[i][0].split(":")
            sample_i = record[0]
            treatment_i = record[1]
            treatments = treatment_i.split("\n")
            for t in treatments:
                if (t != ""):
                    current_time = datetime.datetime.now()
                    str_time = current_time.strftime('%Y%m%d%H%M%S%f')[:-3]
                    TreatmentID = "{0}".format(str_time)
                    micorbe_name = t.split("+")[0]
                    abundance = t.split("+")[1]
                    print('micorbe_name', micorbe_name, 'abundance', abundance)
                    cursor = c.execute(
                        '''
                        insert into
                        DieaseTreatment
                        values(?,?,?,?)
                        ''', [micorbe_name, abundance, sample_i, HistoryID])
                    conn.commit()
        conn.close()

    return jsonify(response)


@app.route('/DiagnosizeHistory', methods=['GET', 'POST'])
def DiagnosizeHistory():
    record_list = []
    conn = sqlite3.connect(DiagnosisHistory_path)
    c = conn.cursor()
    cursor = c.execute('''
        SELECT *
        FROM DiagnosisHistory
            ''')
    cursors = cursor.fetchall()
    for record in cursors:
        tmp_list = list(record)
        for i in range(len(record)):
            tmp_list[i] = str(tmp_list[i])
        record_list.append(tmp_list)
    # savedict_tofile(path, record, data_dict)
    conn.close()
    response = {'record': record_list}
    return jsonify(response)


@app.route('/SpecificDiagnosize', methods=['GET', 'POST'])
def SpecificDiagnosize():
    record_list = []
    if (request.method == "GET"):
        print('got it')

    if request.method == 'POST':  # 获取vue中传递的值
        data = request.get_data()
        data = json.loads(data)
        Diagnosizeid = data['Diagnosizeid']
        print("Diagnosizeid", Diagnosizeid)
        conn = sqlite3.connect(DiagnosisHistory_path)
        c = conn.cursor()
        cursor = c.execute(f'''
            SELECT *
            FROM SpecficDiagnosis
            where HistoryID = {str(Diagnosizeid)}''')
        cursors = cursor.fetchall()
        for record in cursors:
            tmp_list = list(record)
            for i in range(len(record)):
                tmp_list[i] = str(tmp_list[i])
            record_list.append(tmp_list)
        # savedict_tofile(path, record, data_dict)
        conn.close()
    response = {'record': record_list}
    return jsonify(response)


@app.route('/DieaseTreatment', methods=['GET', 'POST'])
def DieaseTreatment():
    record_list = []
    if (request.method == "GET"):
        print('got it')

    if request.method == 'POST':  # 获取vue中传递的值
        data = request.get_data()
        data = json.loads(data)
        sampleID = data['sampleID']
        historyID = data['historyID']
        print(sampleID, historyID)
        conn = sqlite3.connect(DiagnosisHistory_path)
        c = conn.cursor()
        cursor = c.execute(
            f'''
            SELECT *
            FROM DieaseTreatment
            where SpecificID = ? and HistoryID = ? ''',
            [sampleID.lower(), historyID])
        cursors = cursor.fetchall()
        for record in cursors:
            tmp_list = list(record)
            for i in range(len(record)):
                tmp_list[i] = str(tmp_list[i])
            record_list.append(tmp_list)
        print('Treatment', record_list)
        # savedict_tofile(path, record, data_dict)
        conn.close()
    response = {'record': record_list[1:len(record_list) - 1]}
    return jsonify(response)


# 启动运行
if __name__ == '__main__':
    app.run()  # 这样子会直接运行在本地服务器，也即是 localhost:5000
# app.run(host='your_ip_address') # 这里可通过 host 指定在公网IP上运行
