from adat.dataset import Datasets, Dataset
import pandas as pd

from adat.paths import HERE

df = pd.read_csv(HERE.joinpath('../tests/adult_csv.csv'))

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'native-country']
cont_names = ['age', 'education-num', 'capitalloss', 'capitalgain', 'hoursperweek']

all_cols = cat_names + cont_names

t = ['age', 'race']
x = [e for e in all_cols if e not in t]

undes = ['fnlwgt']
if 'fnlwgt' in undes:
    df = df.drop(columns='fnlwgt')

t_df = df[[c for c in all_cols if c in t]]
t_cat_cols = [c for c in cat_names if c in t]
t_cont_cols = [c for c in cont_names if c in t]

x_df = df[[c for c in all_cols if c not in t]]
x_cat_cols = [c for c in cat_names if c not in t]
x_cont_cols = [c for c in cont_names if c not in t]

# %%

datas = Dataset(name='adult_census_data',
                descr='census dataset',
                type='real',
                categ_cols=cat_names,
                cont_cols=cont_names,
                x_cols=x,
                t_cols=t,
                sample=df.sample(10),
                cols_to_remove=['fnlwgt'],
                df=df)

#%%
Datasets.set_gcp_creds(HERE.joinpath('../tests/secrets.json').as_posix())
Datasets.save(datas)

#%%
data_lst = Datasets.list()

#%%
datas = Datasets.load(data_lst[0])
