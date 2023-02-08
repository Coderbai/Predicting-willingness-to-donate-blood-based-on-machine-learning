import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import math
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.decomposition import PCA
import xgboost as xgb
import datetime
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import sklearn
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, roc_curve
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
    # print('index_',index_)
    # print('target_inputs',target_inputs)
    # inputs = np.vstack((source_inputs, target_inputs[index_, :]))
    # labels = np.append(source_labels, target_labels[index_])
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
        for i, col_ in enumerate(['置信度']+data_col):
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

files = os.listdir('./data/exp_data/yangzhou2/data/')
files.sort()
data = None
label = None
for file in files:
    if not '.npz' in file:
        continue
    a = np.load('./data/exp_data/yangzhou2/data/'+file)
    data_, label_ = a['arr_0'], a['arr_1']
    print(file, data_.shape[0])
    print('data_, label_', data_.shape, label_.shape)
    print('data',data)
    if data is None and data_.shape[0]!=0:
        data = data_
        label = label_
        print(type(label))
        print(type(data))
    elif data is not None and data_.shape[0]!=0:
        print('data, data_', data.shape, data_.shape)
        data = np.vstack((data, data_))
        label = np.append(label, label_)
print('data.shape', data.shape)
print('label',label,label.shape)
print(label.sum())
interval = data[:7]
max_interval = np.max(interval)
min_interval = np.min(interval)
mean_interval = np.mean(interval)
var_interval = np.var(interval)
std_interval = np.std(interval)
print(max_interval, min_interval, mean_interval, var_interval, std_interval)
print(data.shape)
x_train, y_train, x_test, y_test = data_split(data, label)
print('len(y_train)',len(y_train),len(x_train))
print('len(y_test)',len(y_test),len(x_test))
# print('len(y_val)',len(y_val),len(x_val))
print(y_train.sum()/len(y_train))
# print(y_val.sum()/len(y_val))
print(y_test.sum()/len(y_test))

# f = open('aucdata-finetune.csv','a+')
f = open('aucdata-6feat.csv','a+')
# f = open('aucdata-pca6feat.csv','a+')
data = data[:,[7,0,11,3,8,6]]
# pca = PCA(n_components=6)
# pca.fit(data)
# data = pca.fit_transform(data)
print('selected data shape', data.shape)
accuracy_train, recall, precision, auc, f1 = [],[],[],[],[]
fpr_forplot, tpr_forplot, auc_forplot = [], [], []

# hyperparameter finetune
# param_grid = {'max_depth': [6, 15, 30, 50, None], 'min_samples_split': [1,2,5,10], 'min_samples_leaf': [1,2,5,10]}
# param_grid = {'C':[0.0001,1,100,10000],'gamma':[0.1,0.5,1.0]}
# model = XGBClassifier(penalty='l2')
# model = RandomForestClassifier()
# model = sklearn.svm.SVC(probability=True)

# model = MLPClassifier()
# param_grid = {'alpha':[0.0001, 0.01, 0.001, 0.1], 'activation':['relu','logistic','tanh'], 'solver':['adam','sgd']}
# model = LogisticRegression(penalty='l2')
# param_grid = {'penalty':['l1', 'l2'], 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag']}
# grid = GridSearchCV(model, param_grid, cv=10, scoring='f1')
# grid.fit(data, label)
# print(grid.best_params_)
# print('=========')

for i in range(10):
    # model = XGBClassifier(penalty='l2', max_depth=50, min_child_weight=1.5)
    model = XGBClassifier(penalty='l2')
    x_train, y_train, x_test, y_test = data_split(data, label)
    model.fit(x_train, y_train)
    predict_test = model.predict(x_test)
    # # Accuray Score on train dataset
    accuracy_train.append(accuracy_score(y_test, predict_test))
    recall.append(recall_score(y_test, predict_test))
    precision.append(precision_score(y_test, predict_test))
    auc.append(roc_auc_score(y_test, predict_test))
    f1.append(f1_score(y_test, predict_test))
    print(y_test.shape, predict_test.shape)
    pred_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    auc_forplot.append(auc)
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC of XGBoost')
    plt.plot(fpr, tpr, 'b', label='Val AUC = 0.809')
    plt.legend(loc='lower right')
    plt.grid()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

n, min_max, mean, var, skew, kurt = stats.describe(accuracy_train)
print('accuracy', accuracy_train)
n, min_max, mean, var, skew, kurt = stats.describe(recall)
print('recall', recall)
n, min_max, mean, var, skew, kurt = stats.describe(precision)
print('precision', precision)
n, min_max, mean, var, skew, kurt = stats.describe(auc)
print('auc', auc)
n, min_max, mean, var, skew, kurt = stats.describe(f1)
print('f1', f1)
res = np.array([accuracy_train,recall,precision,auc,f1])
np.savetxt(f, res, delimiter=",")


accuracy_train, recall, precision, auc, f1 = [],[],[],[],[]
for i in range(10):
    model = RandomForestClassifier(max_depth=None, min_samples_split=1.0, min_samples_leaf=2)
    x_train, y_train, x_test, y_test = data_split(data, label)
    model.fit(x_train, y_train)
    predict_test = model.predict(x_test)
    # # Accuray Score on train dataset
    accuracy_train.append(accuracy_score(y_test, predict_test))
    recall.append(recall_score(y_test, predict_test))
    precision.append(precision_score(y_test, predict_test))
    auc.append(roc_auc_score(y_test, predict_test))
    f1.append(f1_score(y_test, predict_test))
    pred_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    auc_forplot.append(auc)
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC of RandomForest')
    plt.plot(fpr, tpr, 'b', label='Val AUC = 0.797')
    plt.legend(loc='lower right')
    plt.grid()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
n, min_max, mean, var, skew, kurt = stats.describe(accuracy_train)
print('accuracy', accuracy_train)
n, min_max, mean, var, skew, kurt = stats.describe(recall)
print('recall', recall)
n, min_max, mean, var, skew, kurt = stats.describe(precision)
print('precision', precision)
n, min_max, mean, var, skew, kurt = stats.describe(auc)
print('auc', auc)
n, min_max, mean, var, skew, kurt = stats.describe(f1)
print('f1', f1)
res = np.array([accuracy_train,recall,precision,auc,f1])
np.savetxt(f, res, delimiter=",")



accuracy_train, recall, precision, auc, f1 = [],[],[],[],[]
for i in range(10):
    # model = sklearn.svm.SVC(probability=True, C=10000, gamma=0.1)
    model = sklearn.svm.SVC(probability=True)
    x_train, y_train, x_test, y_test = data_split(data, label)
    model.fit(x_train, y_train)
    predict_test = model.predict(x_test)
    # # Accuray Score on train dataset
    accuracy_train.append(accuracy_score(y_test, predict_test))
    recall.append(recall_score(y_test, predict_test))
    precision.append(precision_score(y_test, predict_test))
    auc.append(roc_auc_score(y_test, predict_test))
    f1.append(f1_score(y_test, predict_test))
    print('1')
    pred_prob = model.predict_proba(x_test, )[:, 1]
    print('2')
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    print('3')
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    print('4')
    auc_forplot.append(auc)
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC of SVM')
    plt.plot(fpr, tpr, 'b', label='Val AUC = 0.552')
    plt.legend(loc='lower right')
    plt.grid()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
n, min_max, mean, var, skew, kurt = stats.describe(accuracy_train)
print('accuracy', accuracy_train)
n, min_max, mean, var, skew, kurt = stats.describe(recall)
print('recall', recall)
n, min_max, mean, var, skew, kurt = stats.describe(precision)
print('precision', precision)
n, min_max, mean, var, skew, kurt = stats.describe(auc)
print('auc', auc)
n, min_max, mean, var, skew, kurt = stats.describe(f1)
print('f1', f1)
res = np.array([accuracy_train,recall,precision,auc,f1])
np.savetxt(f, res, delimiter=",")

accuracy_train, recall, precision, auc, f1 = [],[],[],[],[]
for i in range(10):
    # model = MLPClassifier(activation='relu', alpha=0.0001, solver='adam')
    model = MLPClassifier()
    x_train, y_train, x_test, y_test = data_split(data, label)
    model.fit(x_train, y_train)
    predict_test = model.predict(x_test)
    # # Accuray Score on train dataset
    accuracy_train.append(accuracy_score(y_test, predict_test))
    recall.append(recall_score(y_test, predict_test))
    precision.append(precision_score(y_test, predict_test))
    auc.append(roc_auc_score(y_test, predict_test))
    f1.append(f1_score(y_test, predict_test))
    pred_prob = model.predict_proba(x_test, )[:, 1]
    print('2')
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    print('3')
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    print('4')
    auc_forplot.append(auc)
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC of DNN')
    plt.plot(fpr, tpr, 'b', label='Val AUC = 0.666')
    plt.legend(loc='lower right')
    plt.grid()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
n, min_max, mean, var, skew, kurt = stats.describe(accuracy_train)
print('accuracy', accuracy_train)
n, min_max, mean, var, skew, kurt = stats.describe(recall)
print('recall', recall)
n, min_max, mean, var, skew, kurt = stats.describe(precision)
print('precision', precision)
n, min_max, mean, var, skew, kurt = stats.describe(auc)
print('auc', auc)
n, min_max, mean, var, skew, kurt = stats.describe(f1)
print('f1', f1)
res = np.array([accuracy_train,recall,precision,auc,f1])
np.savetxt(f, res, delimiter=",")

accuracy_train, recall, precision, auc, f1 = [],[],[],[],[]
for i in range(10):
    model = KNeighborsClassifier()
    x_train, y_train, x_test, y_test = data_split(data, label)
    model.fit(x_train, y_train)
    predict_test = model.predict(x_test)
    # # Accuray Score on train dataset
    accuracy_train.append(accuracy_score(y_test, predict_test))
    recall.append(recall_score(y_test, predict_test))
    precision.append(precision_score(y_test, predict_test))
    auc.append(roc_auc_score(y_test, predict_test))
    f1.append(f1_score(y_test, predict_test))
    pred_prob = model.predict_proba(x_test, )[:, 1]
    print('2')
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    print('3')
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    print('4')
    auc_forplot.append(auc)
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC of KNN')
    plt.plot(fpr, tpr, 'b', label='Val AUC = 0.645')
    plt.legend(loc='lower right')
    plt.grid()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
n, min_max, mean, var, skew, kurt = stats.describe(accuracy_train)
print('accuracy', accuracy_train)
n, min_max, mean, var, skew, kurt = stats.describe(recall)
print('recall', recall)
n, min_max, mean, var, skew, kurt = stats.describe(precision)
print('precision', precision)
n, min_max, mean, var, skew, kurt = stats.describe(auc)
print('auc', auc)
n, min_max, mean, var, skew, kurt = stats.describe(f1)
print('f1', f1)
res = np.array([accuracy_train,recall,precision,auc,f1])
np.savetxt(f, res, delimiter=",")

accuracy_train, recall, precision, auc, f1 = [],[],[],[],[]
for i in range(10):
    model = tree.DecisionTreeClassifier()
    x_train, y_train, x_test, y_test = data_split(data, label)
    model.fit(x_train, y_train)
    predict_test = model.predict(x_test)
    # # Accuray Score on train dataset
    accuracy_train.append(accuracy_score(y_test, predict_test))
    recall.append(recall_score(y_test, predict_test))
    precision.append(precision_score(y_test, predict_test))
    auc.append(roc_auc_score(y_test, predict_test))
    f1.append(f1_score(y_test, predict_test))
    pred_prob = model.predict_proba(x_test, )[:, 1]
    print('2')
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    print('3')
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    print('4')
    auc_forplot.append(auc)
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC of Decision Tree')
    plt.plot(fpr, tpr, 'b', label='Val AUC = 0.753')
    plt.legend(loc='lower right')
    plt.grid()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
n, min_max, mean, var, skew, kurt = stats.describe(accuracy_train)
print('accuracy', accuracy_train)
n, min_max, mean, var, skew, kurt = stats.describe(recall)
print('recall', recall)
n, min_max, mean, var, skew, kurt = stats.describe(precision)
print('precision', precision)
n, min_max, mean, var, skew, kurt = stats.describe(auc)
print('auc', auc)
n, min_max, mean, var, skew, kurt = stats.describe(f1)
print('f1', f1)
res = np.array([accuracy_train,recall,precision,auc,f1])
np.savetxt(f, res, delimiter=",")


accuracy_train, recall, precision, auc, f1 = [],[],[],[],[]
for i in range(10):
    # model = LogisticRegression(penalty='l1', solver='liblinear')
    model = LogisticRegression()
    x_train, y_train, x_test, y_test = data_split(data, label)
    model.fit(x_train, y_train)
    predict_test = model.predict(x_test)
    # # Accuray Score on train dataset
    accuracy_train.append(accuracy_score(y_test, predict_test))
    recall.append(recall_score(y_test, predict_test))
    precision.append(precision_score(y_test, predict_test))
    auc.append(roc_auc_score(y_test, predict_test))
    f1.append(f1_score(y_test, predict_test))
    pred_prob = model.predict_proba(x_test, )[:, 1]
    print('2')
    fpr, tpr, threshold = roc_curve(y_test, pred_prob)
    print('3')
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    print('4')
    auc_forplot.append(auc)
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC of LR')
    plt.plot(fpr, tpr, 'b', label='Val AUC = 0.687')
    plt.legend(loc='lower right')
    plt.grid()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
n, min_max, mean, var, skew, kurt = stats.describe(accuracy_train)
print('accuracy', accuracy_train)
n, min_max, mean, var, skew, kurt = stats.describe(recall)
print('recall', recall)
n, min_max, mean, var, skew, kurt = stats.describe(precision)
print('precision', precision)
n, min_max, mean, var, skew, kurt = stats.describe(auc)
print('auc', auc)
n, min_max, mean, var, skew, kurt = stats.describe(f1)
print('f1', f1)
res = np.array([accuracy_train,recall,precision,auc,f1])
np.savetxt(f, res, delimiter=",")

