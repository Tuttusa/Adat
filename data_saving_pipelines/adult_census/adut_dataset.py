from adat.dataset import Datasets, Dataset
import pandas as pd

from adat.paths import HERE

df = pd.read_csv(HERE.joinpath('../data_saving_pipelines/adult_census/adult_census_data.csv'))

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
             'native-country', 'income']
cont_names = ['age', 'educational-num', 'capital-loss', 'capital-gain', 'hours-per-week']

all_cols = cat_names + cont_names

t = ['age', 'race']
y = 'income'
x = [e for e in all_cols if e not in t and e != y]

t_df = df[[c for c in all_cols if c in t and c != y]]
t_cat_cols = [c for c in cat_names if c in t and c != y]
t_cont_cols = [c for c in cont_names if c in t and c != y]

x_df = df[[c for c in all_cols if c not in t and c != y]]
x_cat_cols = [c for c in cat_names if c not in t and c != y]
x_cont_cols = [c for c in cont_names if c not in t and c != y]

# %%

datas = Dataset(name='adult_census_data',
                descr='census dataset',
                type='real',
                categ_cols=cat_names,
                cont_cols=cont_names,
                x_cols=x,
                t_cols=t,
                y_cols=y,
                sample=df.sample(10),
                cols_to_remove=['fnlwgt'],
                df=df)

#%%
prepross = datas.transform(datas.df)
dff = datas.inverse_transform(prepross)

#%%
Datasets.set_gcp_creds(HERE.joinpath('../.secrets/secrets.json').as_posix())
Datasets.save(datas)

#%%
data_lst = Datasets.list()

#%%
datas = Datasets.load(data_lst[0])
