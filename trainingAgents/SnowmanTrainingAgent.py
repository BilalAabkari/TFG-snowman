import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Fh


torch.manual_seed(13)


from utils.ReplayMemory import Transition

class TrainingAgent:
    def __init__(self, policy_net, target_net, optimizer, replay_memory, eps_end, eps_start, eps_decay, gamma, batch_size, initial_random_steps, device):
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.initial_random_steps = initial_random_steps

        self.policy_net = policy_net
        self.target_net = target_net
        self.replay_memory = replay_memory
        self.steps_done = 0
        self.optimizer = optimizer

    
    def select_action_epsilon_greedy(self, state, environment):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * max((self.steps_done-self.initial_random_steps),0) / self.eps_decay)
        self.steps_done = self.steps_done+1
        if sample > eps_threshold and self.steps_done > self.initial_random_steps:
            with torch.no_grad():
                return False, self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return True, torch.tensor([[environment.action_space.sample()]], device=self.device, dtype=torch.long)
        
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
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #Calculem els Q-valors del model online de cada parella estat-accio. Per a cada estat només ens interessa les accions que s'han realitzat,
        #per tant els obtenim amb gather:
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        #Calculem els Q-valors dels estats següents que calcula la nostra xarxa. Ho fem amb la target net (variant DQN WITH FIXED Q-VALUE TARGETS)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            #L'unic que fa el seguent es que si es un estat final es queda tal qual (q-valor = 0). Si no ho és, guardem els q-valors esperats.
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

            #Calculem els valors esperats segons la fòrmula:
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        #Finalment calculem la pèrdua a partir dels q-valors que ens ha predit la xarxa i els q-valors esperats segons la fòrmula am Huber Loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        #Capem els valors dels gradients per evitar el exploding gradient
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        #self.policy_net.reset_noise()
        #self.target_net.reset_noise()