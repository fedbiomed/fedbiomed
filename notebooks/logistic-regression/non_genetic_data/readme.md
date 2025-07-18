## Step.1 Download category information table from UK biobank website (Data-fields group)
1. Open the script `ukbb_spider.py` to true this flag and add the category IDs and names into the dict object `category_map`;
``` python
if True:
	category_map = {1001:'Primary_demographics', 704:'Lifestyle', 706:'Physical_measures'}
	Category(category_map)
```

2. Runing the script to download the category tables from UKBB.
``` bash
python ukbb_spider.py
```
3. The default saving dir is <u>'./category_DataField_tables'</u> under the working path.  
As the above example, the three files will be generated:  
> 1001_Primary_demographics.txt  
704_Lifestyle.txt  
706_Physical_measures.txt  

*****************************

## Step.2 Mapping table for the information of data-field, data-coding, data type and the columns' index in `ukbxxxx.csv` file (the tabular data with a large size; `xxxx` is a series of numbers specified by the user's UKBB project)
1. Extract the table element from `ukbxxxx.html`;
> open .html file on the browser in source code view (developer), select and copy the table element of the dataset's (ukbxxxx.csv) columns information -> saving as `ukbxxxx_maintable.html`

2. Open the script `ukbb_html_table_process.py` to modify the relevant file path; 
``` python
    ## get mapping table
    df = pd.read_html('ukb41910_maintable.html')[0]
    df = df.iloc[1:] # remove eid row
    mapping_table = sorted_link_table(df)
    
    ## get field IDs and field names (to be extracted)
    categoryTable_path = './category_DataField_tables'
    category_map = { 1001:'Primary_demographics', 704:'Lifestyle', 706:'Physical_measures'}
    category_ID = 704 # modify here for different categories in UKBB
```

3. Runing the script to generate the information table of a category of the data-fields;
``` bash
python ukbb_html_table_process.py
```

4. The default saving dir is the working path.
As the above example, a file for the category of lifestyle will be generated:   
> processed_704_Lifestyle.txt    

5. The columns of the output file:
* *data_field*: a number
* *col_start*: columns' index, to be used to extract data from `ukbxxxx.csv`
* *col_end*: columns' index, to be used to extract data from `ukbxxxx.csv`
* *data_type*: most common-> Integer, Categorical (single), Categorical (multiple), and Continuous
* *data_coding*: a number, 0 means non-coding in UKBB (e.g., age)
* *field_name*: the description of data


*****************************

## Step.3 Extract data from `ukbxxxx.csv` by the column's index
1. Open the script `ukbb_data_extract.py` to modify the relevant file path; 
``` python
    n_participants = 502506 # number of individuals for ukb41910 dataset,including header
    csv_file_path = '/home/comp/ericluzhang/UKBB/ukb41910.csv'  
    targetField_path= 'processed_704_Lifestyle.txt'  
      
    ## saving path
    save_dir = '/tmp/csyuxu/ukbb'
```

2. Runing the script to extact the data 
``` bash
python ukbb_data_extract.py
```

3. The outputs are `xx.npy` file, where `xx` is the column's index of the data in `ukbxxxx.csv` 

*****************************

## Step.4 Download data-coing information from UK Biobank website
1. Open the script `ukbb_spider.py` to true this flag and provide the category info file produced by **Step.2**;
``` python
if True:
	targetField_path= 'processed_1001_Primary_demographics.txt'
	DataCoding(targetField_path)
```

2. Runing the script to download the category tables from UKBB.
``` bash
python ukbb_spider.py
```

3. The default saving dir is <u>'./dataCoding_tables'</u> under the working path.  
The name of output files are the number representing that data-coding in UKBB

*****************************

## Step.5 Factor (Data-field) selection
1. By Mendelian randomization or manually select (data-field IDs are required);  
> `mr_factor.py` is used to select risk factors by p-values from MR causality inference result


2. Get the information of selected factors
> `mr_ukbCateg_overlap.py` (output file: e.g., `t2d_physical_measures.txt`)


*****************************

## Step.6 Data preprocessing and datasets generation
1. The parameter are dependent on users
``` python
disease = 't2d'
category = 'physical_measures'    # primary_demographics, lifestyle, physical_measures
codings_dir = './dataCoding_tables' 
colData_dir = '/tmp/csyuxu/ukbb/data_cols'
save_dir = './data_mat'
eid_path = '/home/comp/csyuxu/PRSIMD/code/icd_revise/eid_after' # the eid file path of the datasets, file name: t2d_train_eid.txt, t2d_val_eid.txt, and t2d_test_eid.txt
eid_idx_path = '/home/comp/csyuxu/PRSIMD/code/icd_revise/0.npy' # the eid data columns from UKBB
``` 
	
2. Runing the script to generate training/test/validation datamatrix for the data-fields from a category.  
``` bash
python ukbb_data_process.py
```

3. The default output path is <u>'./data_mat'</u> under working path.
output file: e.g., `./data_mat/t2d/physical_measures_train_data.txt`

*****************************
This is a brief introduction to extracting the non-genetic risk factors used in this paper. Because the data in UK biobank is not open access, we did not provide examples for running the scripts.