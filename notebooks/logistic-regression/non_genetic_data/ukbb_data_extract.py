#!/usr/bin/python
import csv
import os

import numpy as np
import pandas as pd

# csv.field_size_limit(sys.maxsize)  # python3 do not need this command


def save_data(dir_save, names, data):
    if not os.path.exists(dir_save):  # create file save path
        os.makedirs(dir_save)
    if type(names) is list:  # store to be multi-files
        for i, name in enumerate(names):
            np.save(os.path.join(dir_save, str(name)), data[i])
    else:
        np.save(os.path.join(dir_save, str(names)), data)


def get_data(dir_file, names):
    # Get the data of item names for all individuals
    # return the data list if names is char; a list containing data lists if names is a list
    if type(names) is list:
        data = [[] for i in names]
    else:
        data = []

    inds = names  # get_ind(dir_file, names)
    # remember to check the file encoding, may be error occured by it
    with open(dir_file, "r", encoding="cp936") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for i, row in enumerate(reader):
            if np.mod(i, 50000) == 0:  # precessed steps
                print("Processed data:", i)
            if type(names) is list:
                for j, ind in enumerate(inds):
                    # print(str(len(row))+'\t'+str(ind))
                    data[j].append(row[ind])
            else:
                data.append(row[inds])
    return data


def generate_data(dir_file, dir_save, names):
    names_new = []
    # only generate data which are not generated before
    for name in names:
        if not os.path.isfile(os.path.join(dir_save, str(name)) + ".npy"):
            names_new.append(name)

    # if all items are generated before
    if not names_new:
        return
    data_new = get_data(dir_file, names_new)
    save_data(dir_save, names_new, data_new)


# read multi-columns text file
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


# generate columns index list
# def colsInd_generated(info_file):
# a,b = readbyLines_multi(info_file,header='yes')
# cols = []
# print("extract num of fetures:",range(b))
# for i in range(b):
# start = int(a[i][1])
# offset = int(a[i][2])
# col = np.arange(start,start+offset).tolist()
# cols += col
# return cols


def colsInd_generated(info_file):
    temp = pd.read_table(info_file)
    s = temp["col_start"].values.ravel()
    e = temp["col_end"].values.ravel()
    cols = []
    print("extract num of fetures:", temp.shape[0])
    for i in range(temp.shape[0]):
        start = int(s[i])
        end = int(e[i] + 1)
        col = np.arange(start, end).tolist()
        cols += col
    return cols


# Getting data from UKBB csv file
if __name__ == "__main__":
    n_participants = (
        502506  # number of individuals for ukb41910 dataset,including header
    )
    csv_file_path = "/home/comp/ericluzhang/UKBB/ukb41910.csv"
    # targetField_path= 'processed_704_Lifestyle.txt'
    targetField_path = "pca_dcode_opcs_ethic_info.txt"

    ## saving path
    save_dir = "/tmp/csyuxu/ukbb"
    save_dir1 = os.path.join(save_dir, "data_cols")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(save_dir1):
        os.mkdir(save_dir1)

    ## running
    # cols = colsInd_generated(targetField_path)
    cols = [89, 90, 91, 92]
    generate_data(csv_file_path, save_dir1, cols)
