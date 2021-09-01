to setup this folder with celeba:

Download required files of Celeba dataset from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg), files needed are : `img/img_align_celeba.zip` and `Anno/list_attr_celeba.txt`. 
extract the `img_align_celeba.zip` inside the `Celeba_raw/raw/img_align_celeba` folder
put `list_attr_celeba.txt` inside Celeba_raw/raw

the folder will look like :
```
Celeba
│   README.md
│   create_node_data.py    
│   .gitignore
│
└───celeba_raw
│   └───raw
│       │   list_attr_celeba.txt
│       │   img_align_celeba.zip
│       └───img_align_celeba
|           | lots of images 
```

Run the `create_node_data.py script` inside `data/Celeba`

the dataset will be splited into 3 nodes, each containing a csv linking the name of the file and its target and a folder containing the images
the target is only to determine if the target is smiling, but it can be changed inside the `create_node_data.py`