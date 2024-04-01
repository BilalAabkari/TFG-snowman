import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, layers, n_actions, n, m):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(n*m*layers, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, n_actions*m*n)

    def forward(self, x):
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = F.relu(self.linear1(x))
        #print(x.shape)
        x = F.relu(self.linear2(x))
        #print(x.shape)

        x = self.linear3(x)
        

        return x