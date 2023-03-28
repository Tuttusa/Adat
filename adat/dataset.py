import copy
import pickle
from dataclasses import dataclass
from io import BytesIO

import pandas as pd
from typing import List
from google.cloud import storage
import os

BUCKET_NAME = 'adat_tuttusa'


@dataclass
class Dataset:
    name: str
    descr: str
    type: str
    categ_cols: List[str]
    cont_cols: List[str]

    x_cols: List[str]
    t_cols: List[str]
    sample: pd.DataFrame
    cols_to_remove: List[str]

    df: pd.DataFrame

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
        return f"{self.name}.df"

    def load_df(self) -> None:
        client = storage.Client()

        # Get a bucket reference
        bucket = client.bucket(BUCKET_NAME)

        # Get a blob reference with the given file name
        blob = bucket.blob(self.df_name)

        csv_string = blob.download_as_string()

        # Convert the CSV string to a DataFrame
        self.df = pd.read_csv(BytesIO(csv_string))


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
        return datac

    @classmethod
    def save(self, dataset: Dataset, save_df=True):
        self._save_dataclass(dataset)
        if save_df:
            self._save_pandas_df(dataset)
