import os
import re

import numpy as np
import pandas as pd


## print the list of data types
def get_DataType_list(df):
    data_type_col = df["Type"].values.ravel()
    data_types = list(set(data_type_col))
    return data_types


## col ID, data-field ID, data-coding ID, data types
def sorted_link_table(df):
    # Extract Data-Field ID
    DataField_arrIdx = df["UDI"].values.ravel()
    collect = []
    for i in range(len(DataField_arrIdx)):
        DataField, _ = DataField_arrIdx[i].split("-")
        collect.append(DataField)

    # Extract Data coding
    Description = df["Description"].values.ravel()
    prefix_len = len("data-coding ")
    coding_collect = []
    for i in range(len(Description)):
        coding_num = 0  # 0 means no data-coding used
        obj_match = re.search(r"data-coding \d{1,8}", Description[i])
        if obj_match:
            p1 = obj_match.span()[0]
            p2 = obj_match.span()[1]
            coding_num = Description[i][p1 + prefix_len : p2]
        coding_collect.append(coding_num)

    # new table
    DataField_ID = np.array(collect).astype("int").reshape(-1, 1)
    Columns_Idx = df["Column"].values.reshape(-1, 1)
    Dtype = df["Type"].values.reshape(-1, 1)
    DataCoding = np.array(coding_collect).astype("int").reshape(-1, 1)

    sort_table = np.concatenate((Columns_Idx, DataField_ID, DataCoding, Dtype), axis=1)
    return sort_table


## to get col index
def get_colIdx_byDataField(mapping_table, data_field=None):
    col_indexes = mapping_table[:, 0]
    dataField = mapping_table[:, 1]

    match_items = np.argwhere(dataField == data_field).ravel()
    idx_cols = col_indexes[match_items]
    return idx_cols


## to get data coding
def get_dcoding_byDataField(mapping_table, data_field=None):
    dcoding = mapping_table[:, 2]
    dataField = mapping_table[:, 1]

    match_items = np.argwhere(dataField == data_field).ravel()

    coding = dcoding[match_items]
    coding = list(set(coding))

    if len(coding) > 1:
        print(
            "data type error: unmatched data-coding for one data field, check the selected columns!"
        )
        import sys

        sys.exit()
    else:
        coding = coding[0]
    return coding


## to get data type
def get_dtype_byDataField(mapping_table, data_field=None):
    dtype = mapping_table[:, 3]
    dataField = mapping_table[:, 1]

    match_items = np.argwhere(dataField == data_field).ravel()

    type = dtype[match_items]
    type = list(set(type))
    if len(type) == 0:  ## make sure the data field in the csv file
        return False

    elif len(type) > 1:
        print(
            "data type error: unmatched types for one data field, check the selected columns!"
        )
        import sys

        sys.exit()
    else:
        type = type[0]
        return type


if __name__ == "__main__":
    ## get mapping table
    df = pd.read_html("ukb41910_maintable.html")[0]
    df = df.iloc[1:]  # remove eid row
    mapping_table = sorted_link_table(df)

    ## get field IDs and field names (to be extracted)
    categoryTable_path = (
        "/home/comp/csyuxu/PRSIMD/code/non_genetic_data/category_DataField_tables"
    )

    category_map = {
        1001: "Primary_demographics",
        704: "Lifestyle",
        706: "Physical_measures",
        100069: "Female_specific",
    }
    category_ID = 100069  # modify here for different category in UKBB
    categoryTable = pd.read_table(
        os.path.join(
            categoryTable_path,
            str(category_ID) + "_" + category_map[category_ID] + ".txt",
        )
    )

    # categoryTable = pd.read_table(os.path.join(categoryTable_path, 'pca_dcode_opcs_ethic.txt')) # munally operate to extract specific data-fields

    Field_id = categoryTable["Field ID"].values.ravel().astype("int")
    Field_name = categoryTable["Description"].values.ravel()

    ## generate sorted matrix with essential information for each data field
    collect = []
    for i in range(len(Field_id)):
        dtype = get_dtype_byDataField(mapping_table, data_field=Field_id[i])
        if dtype:
            col = get_colIdx_byDataField(mapping_table, data_field=Field_id[i])
            coding = get_dcoding_byDataField(mapping_table, data_field=Field_id[i])
            collect.append([Field_id[i], col[0], col[-1], dtype, coding, Field_name[i]])
    mat = pd.DataFrame(
        np.array(collect),
        columns=[
            "data_field",
            "col_start",
            "col_end",
            "data_type",
            "data_coding",
            "field_name",
        ],
    )

    ## saving for downstream operation
    mat.to_csv(
        "processed_" + str(category_ID) + "_" + category_map[category_ID] + ".txt",
        sep="\t",
        index=False,
    )

    # mat.to_csv('pca_dcode_opcs_ethic_info.txt',sep='\t',index=False)    # munally operate to extract specific data-fields
