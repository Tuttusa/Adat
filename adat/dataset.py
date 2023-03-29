import copy
import pickle
from dataclasses import dataclass
from io import BytesIO
from feature_engine import encoding as ce
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from sklearn.pipeline import Pipeline
from typing import List, Any
from google.cloud import storage
import os
import pandas as pd

BUCKET_NAME = 'adat_tuttusa'


class CustomPipeline(Pipeline):
    def transform(self, X):
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
        return X

    def inverse_transform(self, X):
        for name, transform in self.steps[:-1][::-1]:
            try:
                X = transform.inverse_transform(X)
            except:
                pass
        return X


@dataclass
class Dataset:
    name: str
    descr: str
    type: str
    categ_cols: List[str]
    cont_cols: List[str]

    x_cols: List[str]
    t_cols: List[str]
    y_cols: str
    sample: pd.DataFrame
    cols_to_remove: List[str]

    df: pd.DataFrame

    preprocessor: Any = None

    def __post_init__(self):
        if self.df is not None:
            self._init_preprocessor()

    def _init_preprocessor(self):
        if self.preprocessor is None:
            for col in self.categ_cols:
                self.df[col] = self.df[col].astype('category')

            self.preprocessor = CustomPipeline([
                ('cat_features',
                 ce.OrdinalEncoder(variables=self.categ_cols, encoding_method='arbitrary', missing_values='ignore')),
                ('imputer_cat', CategoricalImputer(variables=self.categ_cols, ignore_format=True)),
                ('num_features', SklearnTransformerWrapper(StandardScaler(), variables=self.cont_cols)),
                ('imputer_num', MeanMedianImputer(imputation_method='mean', variables=self.cont_cols)),
            ])

            self.preprocessor.fit(self.df)

    @property
    def all_cols(self):
        return self.categ_cols + self.cont_cols

    def _filter_cols(self, source, in_lst, out_lst):
        if self.df is None:
            return None
        return self.df[[c for c in source if ((c in in_lst) and (c not in out_lst))]]

    @property
    def t_df(self):
        return self._filter_cols(self.all_cols, self.t_cols, self.cols_to_remove)

    @property
    def x_df(self):
        return self._filter_cols(self.all_cols, self.x_cols, self.cols_to_remove)

    @property
    def t_cat_cols(self):
        return self._filter_cols(self.categ_cols, self.t_cols, self.cols_to_remove)

    @property
    def t_cont_cols(self):
        return self._filter_cols(self.cont_cols, self.t_cols, self.cols_to_remove)

    @property
    def x_cat_cols(self):
        return self._filter_cols(self.categ_cols, self.x_cols, self.cols_to_remove)

    @property
    def x_cont_cols(self):
        return self._filter_cols(self.cont_cols, self.x_cols, self.cols_to_remove)

    @property
    def df_name(self):
        return f"{self.name}.csv"

    def load_df(self) -> None:
        client = storage.Client()

        # Get a bucket reference
        bucket = client.bucket(BUCKET_NAME)

        # Get a blob reference with the given file name
        blob = bucket.blob(self.df_name)

        csv_string = blob.download_as_string()

        # Convert the CSV string to a DataFrame
        self.df = pd.read_csv(BytesIO(csv_string))

    def transform(self, df: pd.DataFrame):
        if self.preprocessor is not None:
            return pd.DataFrame(
                data=self.preprocessor.transform(df),
                columns=self.categ_cols + self.cont_cols
            )
        else:
            raise Exception("preprocessor not initialized")

    def inverse_transform(self, df: pd.DataFrame):
        if self.preprocessor is not None:
            return self.preprocessor.inverse_transform(df)
        else:
            raise Exception("preprocessor not initialized")

    @property
    def categ_encoder_dict(self):
        return self.preprocessor['cat_features'].encoder_dict_

    @property
    def rev_categ_encoder_dict(self):
        return {kd: {v: k for k, v in d.items()} for kd, d in self.categ_encoder_dict.items()}


class Datasets:

    @staticmethod
    def list():
        # Create a GCS client
        client = storage.Client()

        # Get a bucket reference
        bucket = client.bucket(BUCKET_NAME)

        # List all the objects in the bucket
        blobs = bucket.list_blobs()

        # Filter out the objects that end with ".df"
        filtered_blobs = [blob.name for blob in blobs if not blob.name.endswith(".df")]

        return filtered_blobs

    @classmethod
    def set_gcp_creds(self, path_to_key):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_key

    @classmethod
    def _load_dataclass(self, name) -> Dataset:
        client = storage.Client()

        # Get a bucket reference
        bucket = client.bucket(BUCKET_NAME)

        # Get a blob reference with the given file name
        blob = bucket.blob(name)

        # Download the serialized data from the blob
        serialized_data = blob.download_as_bytes()

        # Convert the serialized data to a MyDataClass object using pickle
        data = pickle.loads(serialized_data)

        data = Dataset(**data.__dict__)

        return data

    @classmethod
    def _save_dataclass(self, datas: Dataset):
        # Convert dataclass to bytes using pickle
        dataset = copy.deepcopy(datas)
        dataset.df = None

        serialized_data = pickle.dumps(dataset)

        # Create a GCS client
        client = storage.Client()

        # Get a bucket reference
        bucket = client.bucket(BUCKET_NAME)

        # Create a blob reference with the given file name
        blob = bucket.blob(dataset.name)

        # Upload the serialized data to the blob
        blob.upload_from_string(serialized_data)

        print(f"Data saved to gs://{BUCKET_NAME}/{dataset.name}")

    @classmethod
    def _save_pandas_df(self, dataset: Dataset):

        # Create a GCS client
        client = storage.Client()

        # Get a bucket reference
        bucket = client.bucket(BUCKET_NAME)

        # Convert the DataFrame to a CSV string
        csv_string = dataset.df.to_csv(index=False)

        # Create a blob object and upload the CSV string
        blob = bucket.blob(dataset.df_name)
        blob.upload_from_string(csv_string, content_type="text/csv")

        print(f"DataFrame saved to gs://{BUCKET_NAME}/{dataset.name}")

    @classmethod
    def load(self, name, load_df=True):
        datac = self._load_dataclass(name)
        if load_df:
            datac.load_df()
            datac._init_preprocessor()
        return datac

    @classmethod
    def save(self, dataset: Dataset, save_df=True):
        self._save_dataclass(dataset)
        if save_df:
            self._save_pandas_df(dataset)
