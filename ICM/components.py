import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, layers, n, m, out_dimensionality):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(layers*n*m, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, out_dimensionality)

    def forward(self, x):
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = F.relu(self.linear1(x))
        #print(x.shape)
        x = F.relu(self.linear2(x))
        #print(x.shape)
        x = self.linear3(x)
        return x


class InverseModel(nn.Module):
    def __init__(self, in_dimensionality, n_actions):
        super(InverseModel, self).__init__()
        self.linear1 = nn.Linear(in_dimensionality*2, 64)
        self.linear2 = nn.Linear(64, n_actions)

    def forward(self, state1, state2):
        x = torch.cat((state1, state2), dim=1)
        #print(x.shape)
        x = F.relu(self.linear1(x))
        #print(x.shape)
        x = F.softmax(self.linear2(x))

        return x
    
class ForwardModel(nn.Module):
    def __init__(self, in_dimensionality, n_actions):
        super(ForwardModel, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.linear1 = nn.Linear(in_dimensionality+n_actions, 64)
        self.linear2 = nn.Linear(64, in_dimensionality)
        self.n_actions = n_actions

    def forward(self, state, action):
        action_ = torch.zeros(action.shape[0], self.n_actions, device=self.dummy_param.device)
        indices = torch.stack((torch.arange(action.shape[0]).to(self.dummy_param.device), action.squeeze()), dim=0) #cuidado, potser sense squeeze

        indices = indices.tolist()
        action_[indices] = 1
        x = torch.cat((state, action_), dim=1)

        #print(x.shape)
        x = F.relu(self.linear1(x))
        #print(x.shape)
        x = self.linear2(x)

        return x
    


def loss_fn(q_loss, inverse_loss, forward_loss, beta, lamda):
    loss_ = (1-beta)*inverse_loss
    loss_ += beta * forward_loss
    loss_ = loss_.sum()/loss_.flatten().shape[0]
    loss = loss_ + lamda * q_loss
    return loss


def ICM(state1, action, state2, encoder, forward_model, inverse_model, inverse_loss, forward_loss, forward_scale=1., inverse_scale=1E-4):
    state1_hat = encoder(state1)
    state2_hat = encoder(state2)
    state2_hat_pred = forward_model(state1_hat.detach(), action.detach())

    forward_pred_error = forward_scale + forward_loss(state2_hat_pred, state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    pred_action = inverse_model(state1_hat, state2_hat)
    inverse_pred_error = inverse_scale * inverse_loss(pred_action, action.detach().flatten()).unsqueeze(dim=1)

    return forward_pred_error, inverse_pred_error