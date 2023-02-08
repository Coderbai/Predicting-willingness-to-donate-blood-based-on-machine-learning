from sklearn.model_selection import train_test_split
# from gcforest.gcforest import GCForest
import os
from sklearn.ensemble import RandomForestClassifier
# from utils import *
import sklearn
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
# from keras.models import load_model
import datetime
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
# from deepctr.layers import custom_objects



data_col = ['身份识别码', '出生年份', '性别', '民族', '血型', '最近一次献血时间', '最近一次献血量', '总献血量', '献血次数',
            '最近献血是否合格', '献血频率', '职业', '初次输血时间', '多长时间未献血', '上次献血地点', '文化程度',
            'Rh血型', '居住类型', '是否有献血反应']



def data_split(inputs, labels):
    pos = np.where(labels == 1)
    neg = np.where(labels == 0)
    print('inputs', inputs.shape)
    pos_inputs = inputs[pos[0], :]
    pos_labels = labels[pos[0]]
    neg_inputs = inputs[neg[0], :]
    neg_labels = labels[neg[0]]
    print('len(pos_labels)',len(pos_labels),'len(neg_labels)',len(neg_labels))
    if len(pos_labels) <= len(neg_labels):
        index_ = np.random.choice(len(neg_labels), len(pos_labels), replace=False)
        source_inputs, target_inputs, source_labels, target_labels = pos_inputs, neg_inputs, pos_labels, neg_labels
    else:
        index_ = np.random.choice(len(pos_labels), len(neg_labels), replace=False)
        source_inputs, target_inputs, source_labels, target_labels = neg_inputs, pos_inputs, neg_labels, pos_labels
    print('inputs',inputs.shape)
    x_train_, x_test_, y_train_, y_test_ = train_test_split(inputs, labels, test_size=0.3)
    # x_train_, x_val_, y_train_, y_val_ = train_test_split(x_train_, y_train_, test_size=0.01, random_state=1)
    return x_train_, y_train_, x_test_, y_test_


def accuracy(outputs, labels):
    correct = np.where(outputs==labels)
    correct = len(correct[0])
    return correct / len(labels)

def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 100, "max_depth": 20, "num_class": 2,
         "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


def write_result(inputs=None, blood_dict=None, file_name='./rank_list/rank_list.xlsx', sheet_name='sheet'):
    writer = pd.ExcelWriter(file_name)

    def gen_pd_data(inp):
        data_ = defaultdict(list)
        for i, col_ in enumerate(['confidence']+data_col):
            for k in inp:
                data_[col_].append(k[i])
        return pd.DataFrame(data_)

    if blood_dict is not None:
        assert isinstance(blood_dict, defaultdict) or isinstance(blood_dict, dict)
        for key in blood_dict.keys():
            gen_pd_data(blood_dict[key]).to_excel(writer, sheet_name=key)

    if inputs is not None:
        assert isinstance(inputs, defaultdict) or isinstance(inputs, dict)
        gen_pd_data(inputs).to_excel(writer, sheet_name=sheet_name)
    writer.save()


def plot_learning_curve(estimator,estimator2,estimator3, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    plot a model's learning curve on certain data
    """
    print('train_sizes',train_sizes)
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=10, n_jobs=1, train_sizes=train_sizes)
    print('train_sizes1', train_sizes)
    train_sizes2, train_scores2, test_scores2 = learning_curve(
        estimator2, X, y, cv=10, n_jobs=1, train_sizes=train_sizes)
    print('train_sizes2', train_sizes2)
    train_sizes3, train_scores3, test_scores3 = learning_curve(
        estimator3, X, y, cv=10, n_jobs=1, train_sizes=train_sizes)
    print('train_sizes3', train_sizes3)
    print(test_scores)
    print(test_scores2)
    print(test_scores3)
    sd,sd2,sd3 = np.std(test_scores, axis=1),np.std(test_scores2, axis=1),np.std(test_scores3, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    print('sd',sd)
    print('sd2', sd2)
    print('sd3', sd3)
    test_scores_mean2 = np.mean(test_scores2, axis=1)
    test_scores_mean3 = np.mean(test_scores3, axis=1)
    print('test_scores_mean', test_scores_mean)
    print('test_scores_mean2', test_scores_mean2)
    print('test_scores_mean3', test_scores_mean3)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_mean2 = np.mean(train_scores2, axis=1)
    train_scores_mean3 = np.mean(train_scores3, axis=1)
    print('train_scores_mean',train_scores_mean)
    print('train_scores_mean2',train_scores_mean2)
    print('train_scores_mean3',train_scores_mean3)

    # test_scores_std = np.std(test_scores, axis=1)
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="XGBoost score, SD = %.4f" % float(sd[4]))
    plt.plot(train_sizes, test_scores_mean2, 'o-', color="r",
             label="RandomForest score, SD = %.4f" % float(sd2[4]))
    plt.plot(train_sizes, test_scores_mean3, 'o-', color="b",
             label="DNN score, SD = %.4f" % float(sd3[4]))
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # plt.plot(train_sizes, 1-train_scores_mean, 'o-', color="g",
    #          label="XGBoost loss, SD = %.4f" % float(sd[4]))
    # plt.plot(train_sizes, 1-train_scores_mean2, 'o-', color="r",
    #          label="RandomForest loss, SD = %.4f" % float(sd2[4]))
    # plt.plot(train_sizes, 1-train_scores_mean2, 'o-', color="b",
    #          label="DNN loss, SD = %.4f" % float(sd3[4]))
    # plt.xlabel("Training examples")
    # plt.ylabel("Loss")
    # i = 0
    # for x, y in zip(train_sizes, test_scores_mean):
    #     plt.text(x, y + 0.3, 'sd = %.0f' % sd[i], ha='center', va='bottom', fontsize=10.5)
    #     i = i + 1
    # i = 0
    # for x, y in zip(train_sizes, test_scores_mean2):
    #     plt.text(x, y + 0.3, '%.0f' % sd2[i], ha='center', va='bottom', fontsize=10.5)
    #     i = i + 1
    # i = 0
    # for x, y in zip(train_sizes, test_scores_mean3):
    #     plt.text(x, y + 0.3, '%.0f' % sd3[i], ha='center', va='bottom', fontsize=10.5)
    #     i = i + 1
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()
    plt.savefig('result.jpg')
    print('plot done!')

files = os.listdir('./data/exp_data/yangzhou2/data/')
files.sort()
data = None
label = None
for file in files:
    if not '.npz' in file:
        continue
    print('file',file)
    # file.replace('.npz','A.npz')
    # print('file', file)
    a = np.load('./data/exp_data/yangzhou2/data/'+file)
    # print(a.files)
    data_, label_ = a['arr_0'], a['arr_1']
    # print(data_,label_)
    print(data_.shape[0])
    if data_.shape[0] == 0:
        continue
    if data is None:
        data = data_
        label = label_
        print(type(label))
        print(type(data))
    else:
        data = np.vstack((data, data_))
        label = np.append(label, label_)
print(data.shape)
print('label',label,label.shape)
print(label.sum())
interval = data[:7]
max_interval = np.max(interval)
min_interval = np.min(interval)
mean_interval = np.mean(interval)
var_interval = np.var(interval)
std_interval = np.std(interval)
print(max_interval, min_interval, mean_interval, var_interval, std_interval)
print('data',data.shape)
x_train, y_train, x_test, y_test = data_split(data, label)
print('len(y_train)',len(y_train),len(x_train))
print('len(y_test)',len(y_test),len(x_test))
# print('len(y_val)',len(y_val),len(x_val))
print(y_train.sum()/len(y_train))
# print(y_val.sum()/len(y_val))
print(y_test.sum()/len(y_test))

# model = XGBClassifier(max_depth=15, learning_rate=0.3, n_estimators=300)
xgbmodel = XGBClassifier(max_depth=30, learning_rate=0.4, n_estimators=50, min_child_weight=1,
                         subsample=0.9, colsample_bytree=1, reg_alpha=0, reg_lambda=0.5, scale_pos_weight=0.8,
                         silent=True, objective='binary:logistic', missing=None, eval_metric='error', gamma=0, n_jobs=-1)
# # xgbmodel.fit(x_train,y_train)
# # print('xgbmodel.feature_importances_',xgbmodel.feature_importances_)
# # xgboost.plot_importance(xgbmodel,max_num_features=12)
# # plt.show()
#
rfmodel = RandomForestClassifier(10, criterion='gini',verbose=0, max_depth=30, n_jobs=-1)
dpmodel = MLPClassifier(solver='sgd',activation = 'relu',max_iter = 50,alpha = 1e-5,
                        hidden_layer_sizes = (100,50),random_state = 1)
svmmodel = sklearn.svm.SVC()
plot_learning_curve(xgbmodel,svmmodel,dpmodel, 'Test Accuracy of Different Models', data, label, ylim=None, cv=6,
                        train_sizes=np.linspace(.1, 1.0, 5))

