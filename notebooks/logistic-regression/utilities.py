#!/usr/bin/python

from collections import defaultdict

import numpy as np
import pyreadr as prr
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

## R language co-with Python ##
## Func:Rdata2text
## @param:   1)target_file:
#            2)target_data:
#            3)path_save: the dir of saving result file, String type
## @return: NULL
## @ 2021/03/23
## @ xuyu

# target_file = ["gw_train","gw_test","gw_vali"]
# target_data = ["pred_train","pred_test","pred_val"]


def Rdata2text_multi(target_file, target_data, path_save):  # lines file
    for i in range(len(target_file)):
        data = prr.read_r(path_save + target_file[i] + ".RData")
        data = data[target_data[i]].values

        outfile = open(path_save + target_file[i], "w")  # same filename
        for j in range(data.shape[0]):
            outfile.write(str(data[j][0]) + "\n")
        outfile.close()


def Rdata2text(target_file, target_data, save_suffix, path):
    data = prr.read_r(path + target_file + ".RData")
    data = data[target_data].values
    outfile = open(path + target_file + save_suffix, "w")  # same filename
    for j in range(data.shape[0]):
        outfile.write(str(data[j][0]) + "\n")
    outfile.close()


##  Single column file read ##
## Func:readbyLines
## @param:   1)filepath:  String type, input data shouldnt have empty
#            2)datatype:  read and change the type
## @return: data ,list type ; data size
## @ 2021/03/23
## @ xuyu
def readbyLines(filepath, datatype="string"):
    inputfile = open(filepath)
    data = []
    for line in inputfile:
        A = line.strip()
        if datatype == "string":
            data.append(str(A))
        elif datatype == "float":
            data.append(float(A))
        elif datatype == "int":
            data.append(int(A))
        else:
            print("datatype not support")
    size = len(data)
    inputfile.close()
    return data, size


# read multi-columns text file
# header = 'yes', remove header
def readbyLines_multi(filepath, header="no"):
    inputfile = open(filepath)
    data = []
    for line in inputfile:
        A = line.strip("\t").split()
        data.append(A)
    # remove header/column name
    if header == "yes":
        data = data[1:]

    size = len(data)
    inputfile.close()
    return data, size


## UKBB data process ##


## multi-instance data combination ##
## Func:read_multi_ins , for UKBB multi-instance features
## @param:   1)start: column index in ukbXXXX.csv , Int type
#            2)n_ins: num of instance/columns offsets , Int type
#            3)path_data: the dir of ukbxxx.csv file, String type
## @return: single feature data in UKBB , numpy array type
## @ 2021/03/23
## @ xuyu
def ins_combination(dict_col):
    col = dict_col[0][1:]
    n_participants = 502506
    for i in range(len(dict_col.keys()) - 1):  # i columns/instances
        for j in range(n_participants - 1):
            if col[j] == "" and dict_col[i + 1][j + 1] != "":
                col[j] = dict_col[i + 1][j + 1]
    return col.reshape(n_participants - 1, 1)  # for next concatenate


def read_multi_ins(path, start, n_ins):
    feature_dict = defaultdict(list)
    for i in range(0, n_ins):
        feature_dict[i] = np.load(path + str(start + i) + ".npy")
    return ins_combination(feature_dict)


#  For disease-code processing . codedependency: proce_miss_ill
#  Return :  a sparse matrix, shape[n_sample, n_ins]
def read_illcode_ins(path, start, n_ins):
    # d_matrix = [[] for i in range(502505)]
    for i in range(0, n_ins):
        temp = np.load(path + str(start + i) + ".npy")[1:]
        temp = temp.reshape(-1, 1)
        temp = proce_miss_ill(temp)
        if i == 0:
            d_matrix = temp
        else:
            d_matrix = np.concatenate((d_matrix, temp), axis=1)
    return d_matrix


## miss value process ##
## Func: proce_miss_num , proce_miss_categ
## @param:   1)col: single feature , shape [n,1], numpy array type
#            2)stra: 'median','most_frequent','mean'......
## @return: processed data without missing value by the strategy, numpy array type
## @ 2021/03/23
## @ xuyu


def proce_miss_num(col, stra="median"):  # col shape [n,1]
    for i in range(len(col)):
        if col[i] == "":
            col[i] = 111111
    imputer = SimpleImputer(missing_values=111111, strategy=stra)
    return imputer.fit_transform(col)


def proce_miss_categ(col, stra="most_frequent"):
    for i in range(len(col)):
        if col[i] == "":
            col[i] = 222222
    col = col.astype("int32")  # prevent some type error
    imputer = SimpleImputer(missing_values=222222, strategy=stra)
    return imputer.fit_transform(col)


def proce_miss_ill(col):
    for i in range(len(col)):
        if col[i] == "":
            col[i] = 333333
    imputer = SimpleImputer(missing_values=333333, strategy="constant", fill_value=0)
    return imputer.fit_transform(col.astype("int32"))


## torch tensor type version ##
## Func: one_hot , for categorical data process.
## intput must INT type , index corresponding with class
## @param:   1)data: single feature , shape [n,1], numpy array type
#            2)n_class:  number of different categorical value
## @return:  shape[n,n_class], numpy array type
## @ 2021/03/23
## @ xuyu
def one_hot(data, n_class):
    if type(data) is list:
        data = np.array(data)

    x = torch.zeros((data.size(1), n_class))

    for i in range(data.size(1)):
        x[i][data[0][i].int()] = 1

    return x


## Func: indexList
## Func: sublist
## @param:   1)
#            2)
## @return:  indexList: a list contain the index of subset data in mainset
#            sublist:  a subset
## @ 2021/03/28
## @ xuyu
def indexList(mainset, subset):
    ind_list = []
    for i in range(len(subset)):
        ind = mainset.index(subset[i])
        ind_list.append(ind)
    return ind_list


def sublist(mainset, indexList, datatype="string"):
    sub = []
    for i in range(len(indexList)):
        data = mainset[indexList[i]]
        if datatype == "string":
            sub.append(str(data))
        elif datatype == "float":
            sub.append(float(data))
        elif datatype == "int":
            sub.append(int(data))
        else:
            print("datatype not support")
    return sub


## Func: in_replace_value
## @param:   1)original, target: replace original value by target value
#            2)
## @return:
## @ 2021/03/29
## @ xuyu
def in_replace_value(ar_data, original, target):
    if type(original) is list:
        for i in range(len(ar_data)):
            for j in range(len(original)):
                if ar_data[i] == original[j]:
                    ar_data[i] = target
    else:
        for i in range(len(ar_data)):
            if ar_data[i] == original:
                ar_data[i] = target
    return ar_data


def ex_replace_value(ar_data, remain, target):
    if type(remain) is list:
        for i in range(len(ar_data)):
            for j in range(len(remain)):
                if ar_data[i] != remain[j]:
                    ar_data[i] = target
    else:
        for i in range(len(ar_data)):
            if ar_data[i] != remain:
                ar_data[i] = target
    return ar_data


## KL-Divergence ##
def kl_loss(q, p):
    log_eps = 1e-8  # to avoid the distance to be infinite
    log_q = torch.log(q + log_eps)
    log_p = torch.log(p + log_eps)
    log_diff = log_q - log_p
    kl = torch.sum(torch.sum(q * log_diff, 1), 0)
    return kl / q.size()[0]


# # 1 score
# def kl_loss_score(q, p):
# log_eps = 1e-8 # to avoid the distance to be infinite
# log_q = torch.log(q + log_eps)
# log_p = torch.log(p + log_eps)
# log_diff = log_q - log_p
# kl = torch.sum(q * log_diff)
# return kl/q.size()[0]


## Func: RPS_Distr
## PRS to Distribution ##
## @param:   1)score: Polygenic score
#            2)label: case/control label
## @return:
## @ 2021/03/29
## @ xuyu
def RPS_Distr(score, label):
    X = np.array(score).reshape(-1, 1)
    y = np.array(label)
    clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    PRS_binary = clf.predict_proba(X)  # here is the distribution for 2 class
    return PRS_binary, clf


def save_txt(path, data):
    f = open(path, "w")
    for s in data:
        f.write(str(s) + "\n")
    f.close()


## for debug and single operation
if __name__ == "__main__":
    main_path = "/home/comp/ericluzhang/xuyu"
    wait = input(
        "1 : .RData single column trans to single line file(ldpred2 auto output)\n\
        2:  (ldpred2 grid/ngrid and inf output for ibd&af diseases)\n"
    )
    if wait == "1":
        trans_queue1 = [
            "gw_train_cad",
            "gw_train_ibd",
            "gw_train_bc",
            "gw_train_af",
            "gw_train_t2d",
        ]
        for i in trans_queue1:
            Rdata2text(i, "pred_train", ".score", main_path + "/data/results/ldpred2/")
        trans_queue2 = [
            "gw_test_cad",
            "gw_test_ibd",
            "gw_test_bc",
            "gw_test_af",
            "gw_test_t2d",
        ]
        for i in trans_queue2:
            Rdata2text(i, "pred_test", ".score", main_path + "/data/results/ldpred2/")
        trans_queue3 = [
            "gw_vali_cad",
            "gw_vali_ibd",
            "gw_vali_bc",
            "gw_vali_af",
            "gw_vali_t2d",
        ]
        for i in trans_queue3:
            Rdata2text(i, "pred_val", ".score", main_path + "/data/results/ldpred2/")
    if wait == "2":
        trans_queue1 = [
            "gw_train_ibd_inf",
            "gw_train_af_inf",
            "gw_train_ibd_grid",
            "gw_train_af_grid",
            "gw_train_ibd_gridn",
            "gw_train_af_gridn",
        ]
        for i in trans_queue1:
            Rdata2text(i, "pred_train", ".score", main_path + "/data/results/ldpred2/")
        trans_queue2 = [
            "gw_test_ibd_inf",
            "gw_test_af_inf",
            "gw_test_ibd_grid",
            "gw_test_af_grid",
            "gw_test_ibd_gridn",
            "gw_test_af_gridn",
        ]
        for i in trans_queue2:
            Rdata2text(i, "pred_test", ".score", main_path + "/data/results/ldpred2/")
        trans_queue3 = [
            "gw_vali_ibd_inf",
            "gw_vali_af_inf",
            "gw_vali_ibd_grid",
            "gw_vali_af_grid",
            "gw_vali_ibd_gridn",
            "gw_vali_af_gridn",
        ]
        for i in trans_queue3:
            Rdata2text(i, "pred_val", ".score", main_path + "/data/results/ldpred2/")
    # readbyLines_multi(r'C:\data',header='yes')
