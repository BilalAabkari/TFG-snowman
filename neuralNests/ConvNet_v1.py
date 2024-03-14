import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, layers, n_actions, n, m):
        super(DQN, self).__init__()
        #conv layers params: in_channels, out_channels, kernel_size, padding=0,
        self.conv1 = nn.Conv2d(layers, 16, 5, padding=2) 
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.linear1 = nn.Linear(32*n*m, 256)
        self.linear2 = nn.Linear(256, n_actions)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = F.relu(self.linear1(x))
        #print(x.shape)
        x = self.linear2(x)
        

        return x