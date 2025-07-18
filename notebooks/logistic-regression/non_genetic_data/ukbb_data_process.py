import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def get_Coding_Items_list(dir, codings):
    collect = []
    for i in codings:
        collect.append(pd.read_table(os.path.join(dir, str(i) + ".txt")))
    if len(collect) > 1:
        df_coding = pd.concat(collect, axis=0)
    else:
        df_coding = collect[0]
    return df_coding


def get_Coding_Items_list_forOneType(type_coding, dir, type="Integer"):
    idx = np.argwhere(type_coding[:, 0] == type).ravel()
    all_coding = list(set(type_coding[idx, 1]) - set([0]))
    if all_coding:
        coding_items_list = get_Coding_Items_list(
            dir, all_coding
        )  # -1: do not know; -3: prefer not to answer --> taking them as missing values
        return coding_items_list
    return None


def maually_check_type_coding():
    df = pd.read_table("processed_704_Lifestyle.txt")
    type_coding = df[["data_type", "data_coding"]].values
    codings_dir = "./dataCoding_tables"

    ## for Integer type
    coding_items_list = get_Coding_Items_list_forOneType(
        type_coding, codings_dir, type="Integer"
    )  # -10: half-> convert the values to 0.5
    print(coding_items_list)

    ## for Categorical (single) type
    # coding_items_list = get_Coding_Items_list_forOneType(type_coding, codings_dir, type='Categorical (single)') # -7: None of above-> as another category
    # print(coding_items_list)

    # ## for Categorical (multiple) type
    # coding_items_list = get_Coding_Items_list_forOneType(type_coding, codings_dir, type='Categorical (multiple)') # -7: None of above-> as another category
    # print(coding_items_list)

    # ## for Continuous type (Date and Time are samw with Continuous)
    # coding_items_list = get_Coding_Items_list_forOneType(type_coding, codings_dir, type='Continuous')
    # print(coding_items_list)


def instances_combination(colData_dir, factor):
    cols_idx = [i for i in range(factor["col_start"], factor["col_end"] + 1)]
    collect = []
    for j in cols_idx:
        collect.append(np.load(os.path.join(colData_dir, str(j) + ".npy"))[1:])

    if factor["data_type"] != "Categorical (multiple)":
        _data = collect[0]
        for k in range(1, len(collect)):
            current_ins = collect[k]
            for v in range(len(collect[k])):
                if (
                    _data[v] == "" and current_ins[v] != ""
                ):  # missing value in a instances
                    _data[v] = current_ins[v]

            # ## debug, fill missings by data from the other instance
            # idxx = np.argwhere(_data=='').ravel()
            # print(len(idxx))
        return _data
    else:
        _data = np.array(collect).T  # rows: participants; cols: instances
        return _data


def _categorical_single_processing(data, coding_list, drop_mode="first"):
    ## missing values
    data[data == ""] = "111111"
    data[data == "-1"] = "111111"
    data[data == "-3"] = "111111"
    data[data == "-4"] = "111111"
    _data = data.astype("float")
    imputer = SimpleImputer(missing_values=111111, strategy="most_frequent")
    _data = imputer.fit_transform(_data.reshape(-1, 1))

    ## categories encoding (one-hot)
    if drop_mode == "first":
        enc = OneHotEncoder(handle_unknown="error", drop="first")
    transformed_data = enc.fit_transform(_data).toarray()
    name_categories = enc.categories_[0]

    # if coding != 0:
    # coding_list = pd.read_table(os.path.join(codings_dir, str(coding)+'.txt'))
    # get corresponding categorical name
    name_categories_string = []
    for i in range(len(name_categories)):
        coding_item = coding_list[coding_list["Coding"] == name_categories[i]]
        name_categories_string.append(name + ":" + coding_item["Meaning"].values[0])

    df_tmp = pd.DataFrame(transformed_data)

    if len(name_categories_string) > 1:
        df_tmp.columns = name_categories_string[1:]
    else:
        df_tmp.columns = name_categories_string

    return df_tmp
    # return transformed_data, name_categories


def _categorical_multiple_processing(data, coding_list):
    ## missing values
    data[data == ""] = "0"
    data = data.astype("float")
    ## options
    _data = np.zeros((data.shape[0], coding_list.shape[0]))
    codings = coding_list["Coding"].values.ravel()
    meanings = coding_list["Meaning"].values.ravel()
    for i in range(len(data)):
        for j in range(len(codings)):
            if codings[j] in data[i]:
                print(meanings[j])
                _data[i, j] = 1

    df_tmp = pd.DataFrame(_data)
    df_tmp.columns = meanings
    return df_tmp


def _integer_processing(data, name):
    ## missing values
    data[data == ""] = "111111"
    data[data == "-1"] = "111111"
    data[data == "-3"] = "111111"
    data[data == "-4"] = "111111"
    data[data == "-10"] = "0.5"
    _data = data.astype("float")
    imputer = SimpleImputer(missing_values=111111, strategy="most_frequent")
    _data = imputer.fit_transform(_data.reshape(-1, 1))
    df_tmp = pd.DataFrame(_data)
    df_tmp.columns = [name]
    return df_tmp


def _continuous_processing(data, name):
    ## missing values
    data[data == ""] = "111111"
    data[data == "-1"] = "111111"
    data[data == "-3"] = "111111"
    data[data == "-4"] = "111111"
    _data = data.astype("float")
    imputer = SimpleImputer(missing_values=111111, strategy="median")
    _data = imputer.fit_transform(_data.reshape(-1, 1))
    _data = np.around(_data, 2)  # avoid float error in numpy operation
    df_tmp = pd.DataFrame(_data)
    df_tmp.columns = [name]

    return df_tmp


if __name__ == "__main__":
    # maually_check_type_coding() # the relationship between data types and data codings

    disease = "cad"
    category = (
        "primary_demographics"  # primary_demographics, lifestyle, physical_measures
    )
    codings_dir = "./dataCoding_tables"
    colData_dir = "/tmp/csyuxu/ukbb/data_cols"
    save_dir = "./data_mat"
    eid_path = "/home/comp/csyuxu/PRSIMD/code/icd_revise/eid_after"
    eid_idx_path = "/home/comp/csyuxu/PRSIMD/code/icd_revise/0.npy"

    ## process data for each factor
    df = pd.read_table(disease + "_" + category + ".txt")
    collect = []
    for factor in df.iterrows():
        factor = factor[1]
        # if True:
        # factor = df.iloc[459,:]

        ## load data
        if factor["col_start"] != factor["col_end"]:  # having multi-instance
            data = instances_combination(colData_dir, factor)
        else:  # single instance
            data = np.load(
                os.path.join(colData_dir, str(factor["col_start"]) + ".npy")
            )[1:]

        ## process data according data type, data coding
        type = factor["data_type"]
        coding = factor["data_coding"]
        name = factor["field_name"]

        if type == "Categorical (single)":
            coding_list = pd.read_table(os.path.join(codings_dir, str(coding) + ".txt"))
            transformed_data = _categorical_single_processing(data, coding_list)

        if type == "Categorical (multiple)":
            coding_list = pd.read_table(os.path.join(codings_dir, str(coding) + ".txt"))
            transformed_data = _categorical_multiple_processing(data, coding_list)

        if type == "Integer":
            transformed_data = _integer_processing(data, name)

        if type == "Continuous":
            transformed_data = _continuous_processing(data, name)

        collect.append(transformed_data)
    mat = pd.concat(collect, axis=1)

    ##  generate datasets
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir_ = os.path.join(save_dir, disease)
    if not os.path.exists(save_dir_):
        os.mkdir(save_dir_)

    eid_idx = np.load(eid_idx_path)[1:].astype("int")
    for _set in ["train", "test", "val"]:
        eid = pd.read_table(
            os.path.join(eid_path, disease + "_" + _set + "_eid.txt"), header=None
        ).values.ravel()
        idx = []
        for i in eid:
            idx.append(np.argwhere(eid_idx == i)[0])
        idx = np.array(idx).ravel()

        tmp = mat.iloc[idx, :]
        tmp.to_csv(
            os.path.join(save_dir_, category + "_" + _set + "_data.txt"),
            sep="\t",
            index=None,
        )
        print(tmp)
