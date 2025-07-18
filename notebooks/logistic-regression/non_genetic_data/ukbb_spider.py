import os

import pandas as pd

# from bs4 import BeautifulSoup
import requests


def get_DataCoding_table(codingID):
    url = "https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id=" + str(codingID)
    page = requests.get(url)
    # soup = BeautifulSoup(page.text, 'html.parser')
    # coding_table_html = soup.find_all('table')[1]

    df = pd.read_html(page.text)[1]
    return df


def get_Category_table(categoryID):
    url = "https://biobank.ndph.ox.ac.uk/ukb/label.cgi?id=" + str(categoryID)
    page = requests.get(url)
    df = pd.read_html(page.text)[0]

    df = df.iloc[:, :2]
    return df


def Category(category_map):
    ## Category (groupping data fields)
    sava_dir = "./category_DataField_tables"
    if not os.path.exists(sava_dir):
        os.mkdir(sava_dir)

    # category_map = {1001:'Primary_demographics', 704:'Lifestyle', 706:'Physical_measures'}
    for categoryID in category_map.keys():
        table = get_Category_table(categoryID)
        table.to_csv(
            os.path.join(
                sava_dir, str(categoryID) + "_" + category_map[categoryID] + ".txt"
            ),
            sep="\t",
            index=False,
        )


def DataCoding(targetField_path):
    sava_dir = "./dataCoding_tables"
    if not os.path.exists(sava_dir):
        os.mkdir(sava_dir)
    # targetField_path= 'processed_704_Lifestyle.txt'
    tmp = pd.read_table(targetField_path)
    datacodingID = tmp["data_coding"].values.ravel()
    datacodingID = list(set(datacodingID))
    for i in datacodingID:
        if i != 0:
            table = get_DataCoding_table(i)
            table.to_csv(os.path.join(sava_dir, str(i) + ".txt"), sep="\t", index=False)


if __name__ == "__main__":
    if False:
        # category_map = {1001:'Primary_demographics', 704:'Lifestyle', 706:'Physical_measures'}
        category_map = {100069: "Female_specific"}
        Category(category_map)

    if True:
        targetField_path = "processed_100069_Female_specific.txt"
        DataCoding(targetField_path)
