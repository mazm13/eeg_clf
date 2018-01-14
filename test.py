import numpy as np
import os
import scipy.io as sio
from scipy.fftpack import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

root_dir = './data/'
dprefix = "datas/sleep_data_row3_"
lprefix = "labels/HypnogramAASM_subject"

test_1222 = './data/datas/test_1222.mat'
train3k = './data/datas/train_1222.mat'

mat = sio.loadmat(train3k)
# data = np.array(mat['final_test'])
# print(data.shape)
# print(mat)
datas = []
for key in mat:
    # datas.append(np.array(mat[key]))
    if key in ['__header__', '__version__', '__globals__']:
        continue
    data = np.array(mat[key])
    datas.append(data)
datas = np.concatenate(datas)

index = np.random.randint(0, len(datas))
data = datas[index][:3000]
plt.plot(data, color='blue')

f1 = fft(data)

path = os.path.join(root_dir, dprefix + str(10))
mat = sio.loadmat(path)
datas = np.array(mat['data']).squeeze()
datas = np.reshape(datas, (len(datas) // 1000, 1000))
data = datas[10]

plt.plot(data, color='red')

f2 = fft(data)

plt.show()

plt.plot(f1)
plt.plot(f2, c='yellow')
plt.show()

'''
exit(0)


label_set = []
for i in range(datas.shape[0]):
    label_set.append(datas[i][-1])

labels = list(set(label_set))
print(labels)
exit(0)

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
for i in range(1,21):
    path = os.path.join(root_dir, lprefix + str(i) + ".txt")
    data = []
    f = open(path, "r")
    for i, line in enumerate(f):
        if i == 0:
            continue
        data.append(int(line))
    labels.append(np.array(data))
label_ = np.concatenate(labels)

print(label_.shape)
ll = []
for i in range(label_.shape[0]):
    ll.append(int(label_[i]))

ll = list(set(ll))
print(ll)
exit(0)

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
'''
