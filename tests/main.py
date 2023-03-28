from adat.dataset import Datasets, Dataset
import pandas as pd

#%% List all the datasets
data_lst = Datasets.list()

adult_dataset_name = data_lst[0]

#%% download the dataset metadata
datas = Datasets.load(adult_dataset_name)

#%% download the whole csv
datas.load_df()

#%% view the dataframe
datas.df.head(10)