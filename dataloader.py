from __future__ import print_function

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

root_dir = './data/'
dprefix = "datas/sleep_data_row3_"
lprefix = "labels/HypnogramAASM_subject"

train3k_path = './data/datas/train_1222.mat'
test3k_path = './data/datas/test_1222.mat'


def load_train3k(shuffled_index):
    mats = sio.loadmat(train3k_path)
    data3k = []
    for key in mats:
        if key in ['__header__', '__version__', '__globals__']:
            continue
        data = np.array(mats[key])
        data3k.append(data)
    data3k = np.concatenate(data3k)

    # shuffle for slicing evaluation dataset
    data3k = data3k[shuffled_index]
    return data3k


# dimensionality reduced from 3k to 1k
def sample(data):
    index = np.arange(1000) * 2
    data = data[:,index]
    return data

class TrainData(Dataset):
    def __init__(self, shuffled_index):
        datas = []
        for i in range(1, 21):
            path = os.path.join(root_dir, dprefix + str(i))
            mat = sio.loadmat(path)
            data = np.array(mat['data']).squeeze()
            lenth = data.shape[0]
            data = np.reshape(data, (int(lenth / 1000), 1000))
            datas.append(data)
        data_ = np.concatenate(datas)

        labels = []
        for i in range(1, 21):
            path = os.path.join(root_dir, lprefix + str(i) + ".txt")
            data = []
            f = open(path, "r")
            for j, line in enumerate(f):
                if j == 0:
                    continue
                data.append(int(line))
            labels.append(np.array(data))
        label_ = np.concatenate(labels)

        idx = np.where(label_ != -1)
        data_ = data_[idx] # 121490*1000
        label_ = label_[idx] # shape with (121490,)

        # add train3k
        data3k = load_train3k(shuffled_index)
        data3k_data = data3k[:,:-1]
        data3k_label = data3k[:,-1] #shape with (19105,)

        data3k_data = sample(data3k_data)

        data_ = np.concatenate([data_, data3k_data])
        label_ = np.concatenate([label_, data3k_label.astype(int)])

        scalar = StandardScaler()
        data_ = scalar.fit_transform(data_)
        self.scalar = scalar

        # keep datas whose labels in [1,2,3]
        idx = np.where((label_ <= 3) & (label_ >= 1))
        data_ = data_[idx]
        label_ = label_[idx]

        idx = np.where(label_ == 3)
        label_[idx] = 0

        self.data = torch.from_numpy(data_).float()
        self.label = torch.from_numpy(label_).long()

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]

    def get_transform(self):
        # return self.pca, self.scalar
        return self.scalar


class EvalData(Dataset):
    def __init__(self, shuffled_index, scalar):
        data3k = load_train3k(shuffled_index)
        data = data3k[:,:-1]
        data = sample(data)
        data = scalar.transform(data)

        data_ = data
        label_ = data3k[:,-1].astype(int)

        idx = np.where((label_ <= 3) & (label_ >= 1))
        data_ = data_[idx]
        label_ = label_[idx]

        idx = np.where(label_ == 3)
        label_[idx] = 0

        self.data = torch.from_numpy(data_).float()
        self.label = torch.from_numpy(label_).long()

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]


def evalDataSet(scalar):
    datas = []
    for i in [20]:
        path = os.path.join(root_dir, dprefix + str(i))
        mat = sio.loadmat(path)
        data = np.array(mat['data']).squeeze()
        lenth = data.shape[0]
        data = np.reshape(data, (lenth / 1000, 1000))
        datas.append(data)
    data_ = np.concatenate(datas)

    labels = []
    for i in [20]:
        path = os.path.join(root_dir, lprefix + str(i) + ".txt")
        data = []
        f = open(path, "r")
        for i, line in enumerate(f):
            if i == 0:
                continue
            data.append(int(line))
        labels.append(np.array(data))
    label_ = np.concatenate(labels)

    idx = np.where((label_ <= 3) & (label_ >= 1))
    data_ = data_[idx]
    data_ = scalar.transform(data_)

    label_ = label_[idx]
    return idx, data_, label_
