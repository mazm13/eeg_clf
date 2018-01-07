from __future__ import print_function

import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader import TrainData, EvalData
from model import Clf

from datetime import datetime
import time


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    arg_lr = 0.001
    lr = arg_lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    data3k_len = 38211
    bound = int(data3k_len * 0.5)
    shuffled_index = np.arange(data3k_len)
    np.random.shuffle(shuffled_index)
    shuffled_train = shuffled_index[:bound]
    shuffled_eval = shuffled_index[bound:]

    train_data = TrainData(shuffled_train)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)

    scalar = train_data.get_transform()
    test_data = EvalData(shuffled_eval, scalar=scalar)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0)

    #print(len(train_loader.dataset))
    #exit(0)

    clf = Clf()
    clf = clf.cuda()

    optimizer = optim.SGD(clf.parameters(), lr=0.001)


    # optimizer = optim.Adam(clf.parameters(), lr=0.0001)


    def train(epoch):
        clf.train()
        adjust_learning_rate(optimizer, epoch)
        for batch_idx, (data_e, label_e) in enumerate(train_loader):
            data_c, label_c = data_e.cuda(), label_e.cuda()
            data, label = Variable(data_c), Variable(label_c)
            pred = clf(data)
            loss = F.nll_loss(pred, label)
            clf.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Epoch:{} Loss: {:.6f}'.format(epoch, loss.data[0]))


    def test():
        clf.eval()
        test_loss = 0
        correct = 0
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            output = clf(data)
            test_loss += F.nll_loss(output, label, size_average=False).data[0]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        if float(correct) / len(test_loader.dataset) > 0.82:
            return True
        else:
            return False


    for _ in range(100):
        train(_)
        if test():
            break

    exit(0)
    # Evaluation
    '''
    clf.eval()
    idx, evalData, evalLabel = evalDataSet(scalar=scalar)
    f = open("ret.txt", "a")

    correct = 0
    for i in range(evalData.shape[0]):
        data = evalData[i]
        data = torch.from_numpy(data)
        data = data.cuda()
        data = torch.unsqueeze(data, 0)
        data = Variable(data)
        output = clf(data)
        pred = output.data.max(1, keepdim=True)[1]
        pred = pred.cpu()[0][0]

        if pred == evalLabel[i]:
            correct += 1
        f.write("{},{},{}\n".format(idx[0][i], evalLabel[i], pred))

    f.write("Test accurary: {}".format(float(correct) / evalData.shape[0]))
    f.close()
    '''