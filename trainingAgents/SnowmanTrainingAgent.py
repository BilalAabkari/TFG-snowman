import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Fh
import numpy as np

from ICM.components import ICM
from ICM.components import loss_fn


torch.manual_seed(13)


from utils.ReplayMemory import Transition

class TrainingAgent:
    def __init__(self, 
                 policy_net, 
                 target_net, 
                 optimizer, 
                 replay_memory, 
                 eps_end, 
                 eps_start, 
                 eps_decay, 
                 gamma, 
                 batch_size,
                 eta,
                 beta,
                 lamda,
                 use_explicit,
                 initial_random_steps, 
                 device,
                 encoder,
                 forward_model,
                 inverse_model,
                 inverse_loss,
                 forward_loss
                 ):
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.initial_random_steps = initial_random_steps
        self.encoder = encoder
        self.forward_model= forward_model
        self.inverse_model= inverse_model
        self.inverse_loss= inverse_loss
        self.forward_loss= forward_loss
        self.eta = eta
        self.beta = beta
        self.lamda = lamda
        self.use_explicit = use_explicit

        self.policy_net = policy_net
        self.target_net = target_net
        self.replay_memory = replay_memory
        self.steps_done = 0
        self.optimizer = optimizer

    
    def select_action_epsilon_greedy(self, state, environment):
        sample = random.random()
        valid_actions, invalid_actions = environment.get_valid_actions()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * max((self.steps_done-self.initial_random_steps),0) / self.eps_decay)
        self.steps_done = self.steps_done+1
        if sample > eps_threshold and self.steps_done > self.initial_random_steps:
            with torch.no_grad():
                actions = self.policy_net(state)
                invalid_actions = torch.tensor(invalid_actions, device=self.device, dtype=torch.long)
                actions[0,invalid_actions]=-100000
                return False, actions.max(1).indices.view(1, 1)
        else:
            return True, torch.tensor([[random.choice(valid_actions)]], device=self.device, dtype=torch.long)
        
    def training_step(self):
        #Només entrenem si tenim suficients experiències al replay buffer
        if len(self.replay_memory) < self.batch_size:
           return
        
        #Agafem un conjunt d'experiències aleatories:
        transitions = self.replay_memory.sample(self.batch_size)

        #Tal i com descriu el llibre, necessitem un array per cada camp (array de estats, array de accions, array de seguents estats)
        #La seguent instrucció ho fa automàticament.
        batch = Transition(*zip(*transitions))

        #No ens interessen els estats finals (quan acaba el joc) per tant els filtrem
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])


        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        

        reward_batch = torch.cat(batch.reward)
        
        forward_pred_error, inverse_pred_error = ICM(state_batch, 
                                                     action_batch, 
                                                     next_state_batch, 
                                                     self.encoder, 
                                                     self.forward_model, 
                                                     self.inverse_model, 
                                                     self.inverse_loss, 
                                                     self.forward_loss,
                                                     self.beta,
                                                     self.lamda)
        
        #print("forward_pred_error",forward_pred_error)
        i_reward = (1./self.eta)*forward_pred_error
        reward = i_reward.clone().detach()

        if self.use_explicit:
            reward = reward_batch + reward
                
        

        #Calculem els Q-valors del model online de cada parella estat-accio. Per a cada estat només ens interessa les accions que s'han realitzat,
        #per tant els obtenim amb gather:
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        #Calculem els Q-valors dels estats següents que calcula la nostra xarxa. Ho fem amb la target net (variant DQN WITH FIXED Q-VALUE TARGETS)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        #L'unic que fa el seguent es que si es un estat final es queda tal qual (q-valor = 0). Si no ho és, guardem els q-valors esperats.
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            

        #Calculem els valors esperats segons la fòrmula:
        expected_state_action_values = (next_state_values * self.gamma) + reward
        
        #Finalment calculem la pèrdua a partir dels q-valors que ens ha predit la xarxa i els q-valors esperats segons la fòrmula am Huber Loss
        criterion = nn.MSELoss()
        loss = 1E5*criterion(state_action_values, expected_state_action_values.unsqueeze(1).clone().detach())
        #print("q_loss", loss)
        

        total_loss = loss_fn(loss, inverse_pred_error, forward_pred_error, self.beta, self.lamda)
        loss_list = (loss.cpu(), forward_pred_error.mean().cpu(), inverse_pred_error.mean().cpu())

        total_loss.backward()

        #Capem els valors dels gradients per evitar el exploding gradient
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        #self.policy_net.reset_noise()
        #self.target_net.reset_noise()

        return loss_list
    

    def training_step_with_replay(self):
        #Només entrenem si tenim suficients experiències al replay buffer
        if len(self.replay_memory) < self.batch_size:
           return
        
        #Agafem un conjunt d'experiències aleatories:
        experiences, indices, weights = self.replay_memory.sample(self.batch_size)

        #Tal i com descriu el llibre, necessitem un array per cada camp (array de estats, array de accions, array de seguents estats)
        #La seguent instrucció ho fa automàticament.
        batch = Transition(*zip(*experiences))

        #No ens interessen els estats finals (quan acaba el joc) per tant els filtrem
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])


        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        

        reward_batch = torch.cat(batch.reward)
        
        forward_pred_error, inverse_pred_error = ICM(state_batch, 
                                                     action_batch, 
                                                     next_state_batch, 
                                                     self.encoder, 
                                                     self.forward_model, 
                                                     self.inverse_model, 
                                                     self.inverse_loss, 
                                                     self.forward_loss,
                                                     self.beta,
                                                     self.lamda)
        
        #print("forward_pred_error",forward_pred_error)
        i_reward = (1./self.eta)*forward_pred_error
        reward = i_reward.clone().detach()

        if self.use_explicit:
            reward = reward_batch + reward
                
        

        #Calculem els Q-valors del model online de cada parella estat-accio. Per a cada estat només ens interessa les accions que s'han realitzat,
        #per tant els obtenim amb gather:
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        #Calculem els Q-valors dels estats següents que calcula la nostra xarxa. Ho fem amb la target net (variant DQN WITH FIXED Q-VALUE TARGETS)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        #L'unic que fa el seguent es que si es un estat final es queda tal qual (q-valor = 0). Si no ho és, guardem els q-valors esperats.
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            

        #Calculem els valors esperats segons la fòrmula:
        expected_state_action_values = (next_state_values * self.gamma) + reward
        
        #Finalment calculem la pèrdua a partir dels q-valors que ens ha predit la xarxa i els q-valors esperats segons la fòrmula am Huber Loss
        criterion = nn.MSELoss()
        loss = 1E5*criterion(state_action_values, expected_state_action_values.unsqueeze(1).clone().detach())
        #print("q_loss", loss)

        td_loss = expected_state_action_values.unsqueeze(1) - state_action_values
        td_loss_cpu = td_loss.cpu()
        td_loss_np = td_loss_cpu.detach().numpy()

        self.replay_memory.update_priorities(indices, np.abs(td_loss_np))
        

        total_loss = loss_fn(loss, inverse_pred_error, forward_pred_error, self.beta, self.lamda)
        loss_list = (loss.cpu(), forward_pred_error.mean().cpu(), inverse_pred_error.mean().cpu())

        total_loss.backward()

        #Capem els valors dels gradients per evitar el exploding gradient
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        #self.policy_net.reset_noise()
        #self.target_net.reset_noise()

        return loss_list