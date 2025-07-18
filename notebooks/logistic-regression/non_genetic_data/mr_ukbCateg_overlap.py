import os

import pandas as pd

disease = "t2d"
path = os.path.join("/home/comp/csyuxu/PRSIMD/code/mr", disease)
selected = pd.read_table(os.path.join(path, "field_name_pval.txt"), header=None).values[
    :, 0
]
ukb_info_mat = pd.read_table(
    "processed_706_Physical_measures.txt"
)  # processed_706_Physical_measures, processed_704_Lifestyle

selected_base = ["21003", "31"]  # age, sex
# selected_female_specific = ['2714','2724','3700','3710','2784','2814']


collect = []
for i in range(len(selected)):
    # for i in range(len(selected_base)):
    # tmp = ukb_info_mat[ukb_info_mat['data_field']==int(selected_base[i])]
    tmp = ukb_info_mat[ukb_info_mat["data_field"] == int(selected[i])]

    if not tmp.empty:
        collect.append(tmp)

df = pd.concat(collect, axis=0)
df = df.drop_duplicates()
df.to_csv(disease + "_physical_measures.txt", sep="\t", index=None)
