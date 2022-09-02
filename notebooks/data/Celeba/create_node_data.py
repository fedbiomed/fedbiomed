import os
import numpy as np
import pandas as pd
import shutil

#celeba folder
parent_dir = "./"
celeba_raw_folder = "Celeba_raw/raw"
img_dir = parent_dir + celeba_raw_folder + '/img_align_celeba/'
out_dir = "./celeba_preprocessed"

#read attribute CSV and only load Smilling column
df = pd.read_csv(parent_dir + celeba_raw_folder + '/list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=['Smiling'])

# data is on the form : 1 if the person is smiling, -1 otherwise. we set all -1 to 0 for the model to train faster
df.loc[df['Smiling'] == -1, 'Smiling'] = 0

#split csv in 3 part
length = len(df)
data_node_1 = df.iloc[:int(length/3)]
data_node_2 = df.iloc[int(length/3):int(length/3) * 2]
data_node_3 = df.iloc[int(length/3) * 2:]

#create folder for each node
if not os.path.exists(out_dir + "/data_node_1"):
    os.makedirs(out_dir + "/data_node_1/data")
if not os.path.exists(out_dir + "/data_node_2"):
    os.makedirs(out_dir + "/data_node_2/data")
if not os.path.exists(out_dir + "/data_node_3"):
    os.makedirs(out_dir + "/data_node_3/data")

#save each node's target CSV to the corect folder
data_node_1.to_csv(out_dir + '/data_node_1/target.csv', sep='\t')
data_node_2.to_csv(out_dir + '/data_node_2/target.csv', sep='\t')
data_node_3.to_csv(out_dir + '/data_node_3/target.csv', sep='\t')

#copy all images of each node in the correct folder
for im in data_node_1.index:
    shutil.copy(img_dir+im, out_dir + "/data_node_1/data/" + im)
print("data for node 1 succesfully created")

for im in data_node_2.index:
    shutil.copy(img_dir+im, out_dir + "/data_node_2/data/" + im)
print("data for node 2 succesfully created")

for im in data_node_3.index:
    shutil.copy(img_dir+im, out_dir + "/data_node_3/data/" + im)
print("data for node 3 succesfully created")
