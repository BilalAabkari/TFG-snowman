import numpy as np
import gym
from gym import spaces
import copy
import os
import random
from .SnowmanConstants import SnowmanConstants 

class SnowmanEnvironment(gym.Env):

    ENCODED_TEXT_MODE = 0
    DECODED_TEXT_MODE = 1
    GRAPHIC_MODE = 2 #futur, encara no implementat

    STEP_BACK_PENALIZATION = 10
    BLOCKED_SNOWBALL_PENALIZATION = 400
    INCORRECT_SNOWMAN_PENALIZATION = 400
    INCORRET_NUMBER_OF_SNOWBALLS_PENALIZATION = 400
    

    def __init__(self, map_file, n, m, stop_when_error=False, stop_when_dumb=False, enable_step_back_optimzation=False, enable_blocked_snowman_optimization=False, enable_snowball_number_optimization=False):
        super(SnowmanEnvironment, self).__init__()
        self.n = n
        self.m = m
        map_layers = 7
        self.map, self.decoded_map = self._read_and_encode_map(map_file, n, m)
        self.original_map = copy.deepcopy(self.map) #Conserve the original map to allow reset
        self.stop_when_error = stop_when_error
        self.stop_when_dumb = stop_when_dumb      
        self.enable_snowball_number_optimization = enable_snowball_number_optimization

        #Search the agent position
        for i in range(n):
            for (j) in range(m):
                if (self.map[i,j] == SnowmanConstants.CHARACTER_ON_SNOW_CELL or 
                    self.map[i,j] == SnowmanConstants.CHARACTER_ON_GRASS_CELL):
                    self.agent_position = (i, j)
                    print("setting agent position to ", i, j)
                    break
        self.original_agent_position = copy.deepcopy(self.agent_position)
        self.previous_agent_position = None

        self.optimize_step_back = enable_step_back_optimzation
        self.optimize_blocked_snowballs = enable_blocked_snowman_optimization

        #Atributs necessaris per gym:
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(n, m), dtype=np.float64)

    def reset(self):
        self.map = copy.deepcopy(self.original_map)
        self.agent_position = copy.deepcopy(self.original_agent_position)
        return self.map, { "Agent position" : self.agent_position}

    def step(self, action): # action = 0 dreta, 1 baix, 2 esquerra, 3 dalt
        a, b = self.agent_position
        is_agent_on_snow = self.map[a,b] == SnowmanConstants.CHARACTER_ON_SNOW_CELL
        
        inc=[[0,1,0,-1],[1,0,-1,0]]
        next_cell=[a+inc[0][action],b+inc[1][action]] #seguent posició segons l'acció a realitzar
        next_of_next_cell=[a+2*inc[0][action],b+2*inc[1][action]] #seguent de la seguent posició segons l'acció a realitzar

        mov=SnowmanConstants.actions[int(self.map[next_cell[0],next_cell[1]])][int(self.map[next_of_next_cell[0],next_of_next_cell[1]])] #busca a la matri d'acions segons el que hi ha a la posició seguent i la seguent de la seguent
        reward=mov[3] # la tercera psosició es la recompensa
        mov=mov[:3] # les tres primeres posicions son que hem de col.locar a la posició del jugador, la posició seguent i la posició seguent de la seguent

        #Ajustem el reward segons algunes optimitzacions 
        reward = self.pre_adjust_reward(reward, next_cell, next_of_next_cell)
        
        for i,aux in enumerate(mov):
            if aux!=None:
                if aux==SnowmanConstants.CHARACTER_LEAVE_CELL:
                    if is_agent_on_snow:
                        f=SnowmanConstants.SNOW_CELL
                    else:
                        f=SnowmanConstants.GRASS_CELL
                else:
                    f=int(aux)

                if i==0:
                    self.map[a,b]=f
                elif i==1:
                    self.map[next_cell[0],next_cell[1]]=f
                else:
                    self.map[next_of_next_cell[0],next_of_next_cell[1]]=f
        if self.map[next_of_next_cell[0],next_of_next_cell[1]] == SnowmanConstants.FULL_SNOW_MAN_CELL:
            done=True
        else:
            done=False

        if self.stop_when_error and not done and reward == SnowmanConstants.error:
            done = True

        if self.stop_when_dumb and not done and reward == SnowmanConstants.tonto:
            done = True
        
        #Si l'agent es mou, actualitzem la posicio
        if mov[0] != None:
            if action == 0:
                b = b + 1
            elif action == 1:
                a = a + 1
            elif action == 2:
                b = b - 1
            elif action == 3:
                a = a - 1
            
            if self.map[a,b] != SnowmanConstants.OUT_OFF_GRID_CELL and self.map[a,b] != SnowmanConstants.WALL_CELL:
                self.previous_agent_position = copy.deepcopy(self.agent_position)
                self.agent_position = (a, b)
                #print(self.agent_position)
            
            reward, critical_done = self.post_adjust_reward(reward)
        
            done = done or critical_done

        return self.map,reward,done, {}    
    
    def randomize_agent_position(self):
        possible_positions = []
        for i in range(self.n):
            for j in range(self.m):
                if self.map[i,j] == SnowmanConstants.GRASS_CELL or self.map[i,j] == SnowmanConstants.SNOW_CELL:
                    possible_positions.append((i,j))

        new_i, new_j = random.choice(possible_positions)
        if self.map[new_i,new_j] == SnowmanConstants.GRASS_CELL:
            self.map[new_i, new_j] = SnowmanConstants.CHARACTER_ON_GRASS_CELL
        else:
            self.map[new_i, new_j] = SnowmanConstants.CHARACTER_ON_SNOW_CELL
        
        current_i, current_j = self.agent_position
        if self.map[current_i, current_j] == SnowmanConstants.CHARACTER_ON_GRASS_CELL:
            self.map[current_i, current_j] = SnowmanConstants.GRASS_CELL
        else:
            self.map[current_i, current_j] = SnowmanConstants.SNOW_CELL
        
        self.agent_position = (new_i, new_j)
        self.previous_agent_position = None
        return self.map, {}

    def pre_adjust_reward(self, reward, next_cell, next_of_next_cell):
        #print("adjusting")
        #print("next_cell",next_cell)
        #print("next_of_next_cell",next_of_next_cell)
        adjusted_reward = reward
        """Mirem si la poscio a la que ens movem es la mateixa que la anterior a la actual, per detectar si estem fent un pas endavant i un enrere"""
        if self.optimize_step_back and self.previous_agent_position != None:
            row, col = self.previous_agent_position
            row2 = next_cell[0]
            col2 = next_cell[1]

            if row == row2 and col == col2 and self.map[row2,col2] != SnowmanConstants.WALL_CELL:
                adjusted_reward = adjusted_reward - self.STEP_BACK_PENALIZATION

        """Mirem si el moviment actual ha causat que alguna bola quedi bloquejada i no es pugui moure"""
        if self.optimize_blocked_snowballs:
            #print("optimize_blocked_snowballs")
            #Mirem si es mou alguna bola de neu mirant si la seguent posicio hi ha alguna bola i la següent de la seguent hi ha herba o new
            snowball_in_next_cell = (self.map[next_cell[0],next_cell[1]] == SnowmanConstants.SMALL_BALL_CELL or 
                                     self.map[next_cell[0],next_cell[1]] == SnowmanConstants.MEDIUM_BALL_CELL)
            
            #print("snowball_in_next_cell",snowball_in_next_cell)

            grass_or_snow_in_next_of_next_cell = (self.map[next_of_next_cell[0],next_of_next_cell[1]] == SnowmanConstants.GRASS_CELL or
                                                  self.map[next_of_next_cell[0],next_of_next_cell[1]] == SnowmanConstants.SNOW_CELL)
            
            #print("grass_or_snow_in_next_of_next_cell",grass_or_snow_in_next_of_next_cell)
            
            if snowball_in_next_cell and grass_or_snow_in_next_of_next_cell:
                #Agafem la posicio de la bola que hem mogut, i mirem en les 4 direccions (dalt, baix, esquerra, dreta) quantes estan bloquejadas (hi ha paret)
                snowball_x = next_of_next_cell[0]
                snowball_y = next_of_next_cell[1]
                

                has_horitzontal_cap = (self.map[snowball_x, snowball_y+1] == SnowmanConstants.WALL_CELL or 
                                       self.map[snowball_x, snowball_y-1] == SnowmanConstants.WALL_CELL or 
                                       (self.map[snowball_x, snowball_y+1] == self.map[next_cell[0],next_cell[1]] and snowball_x != next_cell[0] and snowball_y != next_cell[1]) or 
                                       (self.map[snowball_x, snowball_y-1] == self.map[next_cell[0],next_cell[1]] and snowball_x != next_cell[0] and snowball_y != next_cell[1]))
                has_vertical_cap = (self.map[snowball_x+1, snowball_y] == SnowmanConstants.WALL_CELL or 
                                    self.map[snowball_x-1, snowball_y] == SnowmanConstants.WALL_CELL or
                                    (self.map[snowball_x+1, snowball_y] ==self.map[next_cell[0],next_cell[1]] and snowball_x != next_cell[0] and snowball_y != next_cell[1])or 
                                    (self.map[snowball_x-1, snowball_y] == self.map[next_cell[0],next_cell[1]] and snowball_x != next_cell[0] and snowball_y != next_cell[1]))
                
                #print("snowball_x, snowball_2", snowball_x, snowball_y)
                #print("boolean test",self.map[snowball_x+1, snowball_y])


                #print("has_horitzontal_cap",has_horitzontal_cap)
                #print("has_vertical_cap",has_vertical_cap)

                if has_horitzontal_cap and has_vertical_cap:
                    adjusted_reward = adjusted_reward - self.BLOCKED_SNOWBALL_PENALIZATION
        return adjusted_reward
    
    def post_adjust_reward(self, reward):
        adjustedReward = reward
        critical_done = False
        if self.enable_snowball_number_optimization:
            large_balls = 0
            medium_balls = 0
            small_balls = 0
            medium_on_large_balls = 0
            small_on_medium_balls = 0
            small_on_large_balls = 0

            for i in range(self.n):
                for j in range(self.m):
                    if self.map[i,j] == SnowmanConstants.LARGE_BALL_CELL:
                        large_balls = large_balls+1
                    elif self.map[i,j] == SnowmanConstants.SMALL_BALL_CELL:
                        small_balls = small_balls+1
                    elif self.map[i,j] == SnowmanConstants.MEDIUM_BALL_CELL:
                        medium_balls = medium_balls+1
                    elif self.map[i,j] == SnowmanConstants.MEDIUM_BALL_ON_LARGE_BALL_CELL:
                        medium_on_large_balls = medium_on_large_balls+1
                    elif self.map[i,j] == SnowmanConstants.SMALL_BALL_ON_MEDIUM_BALL_CELL:
                        small_on_medium_balls=small_on_medium_balls+1
                    elif self.map[i,j] == SnowmanConstants.SMALL_BALL_ON_LARGE_BALL_CELL:
                        small_on_large_balls=small_on_large_balls+1
            
            #Ninots incorrectes
            if small_on_large_balls > 0 or small_on_medium_balls > 0:
                adjustedReward = adjustedReward - self.INCORRECT_SNOWMAN_PENALIZATION

            if (medium_on_large_balls > 0 and small_balls == 0 or
                medium_on_large_balls == 0 and large_balls > 0 and (medium_balls+small_balls < 2 or small_balls == 0) or
                medium_on_large_balls == 0 and large_balls == 0 and (medium_balls+small_balls < 3 or small_balls == 0) ):
                adjustedReward = adjustedReward - self.INCORRET_NUMBER_OF_SNOWBALLS_PENALIZATION
                critical_done = True

        return adjustedReward, critical_done


            




    def _read_and_encode_map(self,file,n,m):
        print("n,m: " ,n,m)
        tokens='x#,\'.qp1234567'
        replace_tokens=[0,0,8,8,9,11,10,1,2,3,4,5,6,7]
        f = open(file, "r")
        decoded_map = np.empty((n, m), dtype=np.str_)
        encoded_map = np.zeros((n,m))
        for row,linea in enumerate(f):
            linea=linea.rstrip('\n\r\t')
            for column,car in enumerate(linea):
                res = tokens.find(car)
                encoded_map[row,column]=replace_tokens[res]
                decoded_map[row, column] = car
        f.close()
        return encoded_map, decoded_map
    
    def show_map(self, mode):
        if mode == self.ENCODED_TEXT_MODE:
            print(self.map)
        elif mode == self.DECODED_TEXT_MODE:
            tokens = SnowmanConstants.get_cell_codes_array()
            tokens_to_replace = SnowmanConstants.get_tokens_array()
            decoded_map = copy.deepcopy(self.decoded_map)
            for i in range(self.n):
                for j in range(self.m):
                    if self.map[i,j] != SnowmanConstants.WALL_CELL or self.map[i,j] != SnowmanConstants.OUT_OFF_GRID_CELL:
                        index = tokens.index(self.map[i,j])
                        decoded_map[i,j] = tokens_to_replace[index]
            print(decoded_map)

    def split_map_layers(self, state):
        transform=[[0],[4],[5],[4,5],[6],[4,6],[5,6],[4,5,6],[2],[1],[1,3],[2,3]]
        splitted_map=np.zeros((7, self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                for k in transform[int(state[i][j])]:
                    splitted_map[k,i,j]=1
        return splitted_map
    
