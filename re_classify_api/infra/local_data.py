import os
import csv
import numpy
import joblib
import pandas as pd
from scipy import sparse
from gensim.models import Word2Vec

_base_path = 're_classify_api/files/'
_dataset_path = "re_classify_api/files/datasets/Rainbow_Dataset.csv"


def get_dataset():
    return pd.read_csv(_dataset_path, encoding='utf-8')


def get_requirements(file_name):
    return pd.read_csv(_base_path+file_name, encoding='utf-8')['RequirementText']


def get_classes(file_name):
    return pd.read_csv(_base_path+file_name, encoding='utf-8')['Class']


def get_encoded_requirements(file_name):
    return sparse.load_npz(_base_path+file_name)


def get_encoded_classes(file_name):
    return numpy.load(_base_path+file_name)


def get_results(file_name):
    return pd.read_csv("./results/csv/"+file_name, encoding='utf-8')


def write_row(path, file_name, data):
    with open(path+file_name, 'a') as archive:
        writer = csv.writer(archive)
        writer.writerow(data)


def export_to_csv(path, file_name, data_frame):
    data_frame.to_csv(path+file_name, index=False)


def export_to_npy(path, file_name, data):
    numpy.save(path+file_name, data)


def export_to_npz(path, file_name, data):
    sparse.save_npz(path+file_name, data)


def get_word2vec_model(file_name: str):
    return Word2Vec.load(_base_path+file_name)


def export_word2vec_model(model):
    model.save(_base_path+"word2vec_model.model")


def export_sklearn_model(model, file_name):
    joblib.dump(model, _base_path+file_name)


def get_sklearn_model(file_name: str):
    return joblib.load(_base_path+file_name)


def has_file(file_name) -> bool:
    return os.path.isfile(_base_path+file_name)