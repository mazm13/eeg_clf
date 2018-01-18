import torch.nn as nn
import torch.nn.functional as F


class Clf(nn.Module):
    def __init__(self):
        super(Clf, self).__init__()
        self.fc1 = nn.Linear(3000, 1000)
        self.fc2 = nn.Linear(1000, 50)
        self.fc3 = nn.Linear(50, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.fc1(x)
        x = self.fc3(x)

        return F.log_softmax(x)
