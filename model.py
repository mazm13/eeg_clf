import torch.nn as nn
import torch.nn.functional as F


class Clf(nn.Module):
    def __init__(self):
        super(Clf, self).__init__()
        # self.fc1 = nn.Linear(3000, 1000)
        # self.fc2 = nn.Linear(1000, 50)
        # self.fc3 = nn.Linear(50, 3)

        self.conv1 = nn.Conv1d(1, 5, 3, stride=1, padding=1)  # 3000 -> 5 * 3000
        self.bn1 = nn.BatchNorm1d(5)
        self.pool1 = nn.MaxPool1d(10)  # 3000 -> 300
        self.conv2 = nn.Conv1d(5, 3, 3, stride=1, padding=1)  # 300 -> 300
        self.bn2 = nn.BatchNorm1d(3)
        self.pool2 = nn.MaxPool1d(4)  # 300 -> 75
        self.fc = nn.Linear(225, 3)  # 10 -> 3

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc1(x)
        # x = self.fc3(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = self.fc(x)

        return F.log_softmax(x)
