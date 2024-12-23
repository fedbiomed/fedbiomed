To setup this folder with CelebA:

Download required files of Celeba dataset from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg), files needed are : `img/img_align_celeba.zip` and `Anno/list_attr_celeba.txt`. 

Extract the `img_align_celeba.zip` inside the `Celeba_raw/raw` folder (samples will be in `Celeba_raw/raw/img_align_celeba`)
Put `list_attr_celeba.txt` inside Celeba_raw/raw
```
# from $FEDBIOMED_DIR base directory
cd notebooks/data/Celeba
mkdir -p Celeba_raw/raw
cd Celeba_raw/raw
# move img_align_celeba.zip to this directory
unzip img_align_celeba.zip
# move list_attr_celeba.txt to this directory
```

the folder will look like :
```
Celeba
│   README.md
│   create_node_data.py    
│   .gitignore
│
└───Celeba_raw
│   └───raw
│       │   list_attr_celeba.txt
│       │   img_align_celeba.zip
│       └───img_align_celeba
|           | lots of images
```

Preprocess the Celeba data :
```
# from $FEDBIOMED_DIR base directory
# use the python environment for [development](../docs/developer/development-environment.md)
cd notebooks/data/Celeba
python create_node_data.py
```

The dataset will be split into 3 nodes, each containing a csv linking the name of the file and its target and a folder containing the images
The target is only to determine if the target is smiling, but it can be changed inside the `create_node_data.py`.

After running the `create_node_data.py`, the folder should look like:
```
Celeba
│   README.md
│   create_node_data.py    
│   .gitignore
├───Celeba_raw
│   └───raw
│       │   list_attr_celeba.txt
│       │   img_align_celeba.zip
│       └───img_align_celeba
|           └ lots of images
└── celeba_preprocessed
          ├── data_node_1
          │   ├── data
          │   │   └ lots of images
          │   └── target.csv
          ├── data_node_2
          │   ├── data
          │   │   └ lots of images
          │   └── target.csv
          └── data_node_3
              ├── data
              │   └ lots of images
              └── target.csv
```