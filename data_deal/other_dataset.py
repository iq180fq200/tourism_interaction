import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset


def get_distances_parts(src,dst,dataset_path = 'data'):
    dist_data = pd.read_csv(os.path.join(dataset_path, 'dist_edge.csv'),delimiter = '\t',header = None)
    dist_dict = {}
    for idx, item in dist_data.iterrows():
        dist_dict[(item[0],item[1])] = item[2]

    zipped = list(zip(src,dst))
    distances = []
    for item in zipped:
        distances.append(dist_dict[item])

    return np.array(distances).reshape((-1,1))

def get_distances_all(dataset_path = 'data'):
    flow_data = pd.read_csv(os.path.join(dataset_path, 'inter_edge.csv'),delimiter = '\t',header = None)
    dist_data = pd.read_csv(os.path.join(dataset_path, 'dist_edge.csv'),delimiter = '\t',header = None)
    idx_train = np.genfromtxt(os.path.join(dataset_path, 'idx_train.csv')).astype(int)
    idx_valid = np.genfromtxt(os.path.join(dataset_path, 'idx_valid.csv')).astype(int)
    idx_test = np.genfromtxt(os.path.join(dataset_path, 'idx_test.csv')).astype(int)
    dist_dict = {}
    for idx, item in dist_data.iterrows():
        dist_dict[(item[0],item[1])] = item[2]

    train_src = flow_data.iloc[idx_train,0].to_numpy()
    train_dst = flow_data.iloc[idx_train,1].to_numpy()
    zipped = list(zip(train_src,train_dst))
    train_distance = []
    for item in zipped:
        train_distance.append(dist_dict[item])

    valid_src = flow_data.iloc[idx_valid,0].to_numpy()
    valid_dst = flow_data.iloc[idx_valid,1].to_numpy()
    zipped = list(zip(valid_src,valid_dst))
    valid_distance = []
    for item in zipped:
        valid_distance.append(dist_dict[item])

    test_src = flow_data.iloc[idx_test,0].to_numpy()
    test_dst = flow_data.iloc[idx_test,1].to_numpy()
    zipped = list(zip(test_src,test_dst))
    test_distance = []
    for item in zipped:
        test_distance.append(dist_dict[item])

    return np.array(train_distance).reshape((-1,1)),np.array(valid_distance).reshape((-1,1)),np.array(test_distance).reshape((-1,1))

#dataset for batch feed in
class MyDataset(Dataset):
    def __init__(self, x_tensors, y_tensors):
        self.x_tensors = x_tensors
        self.y_tensors = y_tensors

    def __len__(self):
        return len(self.x_tensors)

    def __getitem__(self, idx):
        return self.x_tensors[idx],self.y_tensors[idx]