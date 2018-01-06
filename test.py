import numpy as np
import os
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

root_dir = './data/'
dprefix = "datas/sleep_data_row3_"
lprefix = "labels/HypnogramAASM_subject"

file_ids = range(1, 20)

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
    for i, line in enumerate(f):
        if i == 0:
            continue
        data.append(int(line))
    labels.append(np.array(data))
label_ = np.concatenate(labels)

idx = np.where((label_ <= 3) & (label_ >= 1))
data_ = data_[idx]

pca = PCA(n_components=5)
data = pca.fit_transform(data_)

scalar = StandardScaler()

data = scalar.fit_transform(data)


max_data = np.max(data, axis=0)
min_data = np.min(data, axis=0)
print(max_data)
print(min_data)

#print(m_data.shape)