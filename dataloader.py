from __future__ import print_function

import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

root_dir = './data/'
dprefix = "datas/sleep_data_row3_"
lprefix = "labels/HypnogramAASM_subject"


class BraveData(Dataset):
    def __init__(self, file_ids, pca=None, scalar=None):
        datas = []
        for i in file_ids:
            path = os.path.join(root_dir, dprefix + str(i))
            mat = sio.loadmat(path)
            data = np.array(mat['data']).squeeze()
            lenth = data.shape[0]
            data = np.reshape(data, (lenth / 1000, 1000))
            datas.append(data)
        data_ = np.concatenate(datas)

        labels = []
        for i in file_ids:
            path = os.path.join(root_dir, lprefix + str(i) + ".txt")
            data = []
            f = open(path, "r")
            for j, line in enumerate(f):
                if j == 0:
                    continue
                data.append(int(line))
            labels.append(np.array(data))
        label_ = np.concatenate(labels)

        idx = np.where((label_ <= 3) & (label_ >= 1))

        data_ = data_[idx]
        '''
        if pca is None:
            print("train dataloader")
            pca = PCA(n_components=5)
            data_ = pca.fit_transform(data_)
        else:
            print("test dataloader")
            data_ = pca.transform(data_)
        self.pca = pca
        '''

        if scalar is None:
            scalar = StandardScaler()
            data_ = scalar.fit_transform(data_)
        else:
            data_ = scalar.transform(data_)
        self.scalar = scalar

        self.data = torch.from_numpy(data_)

        label_ = label_[idx]
        idx = np.where(label_ == 3)
        label_[idx] = 0
        self.label = torch.from_numpy(label_)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]

    def get_transform(self):
        # return self.pca, self.scalar
        return self.scalar


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
