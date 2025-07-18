import os

import pandas as pd

disease = "t2d"
path = os.path.join("/home/comp/csyuxu/PRSIMD/code/mr", disease)  # MR dir

# df_1 = pd.read_csv(os.path.join(path, 'features_selection', disease+'_ivw.csv'))
df_1 = pd.read_csv(os.path.join(path, "ivw-" + disease + ".csv"))
df_2 = pd.read_csv(
    os.path.join(path, "features_selection", "features_selection_info.csv"), header=None
).iloc[:, :3]
df_2.columns = ["data_type", "data_field", "field_name"]

df_1 = df_1.drop_duplicates()
selected = df_1[df_1["qval"] < 0.05]

df = selected.sort_values("qval")
df = df.drop_duplicates(subset=["exposure"])
selected = df.iloc[
    :30, :
]  # ranking the factors by q-values in descent and selecting top-30


names = selected["exposure"].values.ravel()
p = selected["qval"].values.ravel()


collect = []
field_ID = []
for s in names:
    a, b = s.split(" || ")
    a = a[:-1]  # remove space char
    collect.append(a)

    df_match = df_2[df_2["field_name"] == str(a)]
    if df_match.shape[0] > 1:
        df_match = df_match.iloc[0, :]
    field_ID.append(int(df_match["data_field"]))


f = open(os.path.join(path, "field_name_pval.txt"), "w")
for i in range(len(collect)):
    f.write(str(field_ID[i]) + "\t" + str(collect[i]) + "\t" + str(p[i]) + "\n")
f.close()
