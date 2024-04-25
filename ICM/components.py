import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

    
class ForwardModel(nn.Module):
    def __init__(self, in_dimensionality, n_actions):
        super(ForwardModel, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.linear1 = nn.Linear(in_dimensionality+n_actions, 128)
        self.linear2 = nn.Linear(128, in_dimensionality)
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
    



def ICM(state1, action, state2, forward_model, forward_loss, forward_scale=1):

    state_1_flattened = torch.flatten(state1, 1)
    state_2_flattened = torch.flatten(state2, 1)

    state2_pred = forward_model(state_1_flattened.detach(), action.detach())

    full_loss=nn.MSELoss()
    forward_full_error = forward_scale * full_loss(state2_pred, state_2_flattened.detach())
    
    #print("real", state_2_flattened[0].cpu().numpy())
    #print("predicted", state2_pred[0].detach().round().cpu().numpy())

    return forward_full_error