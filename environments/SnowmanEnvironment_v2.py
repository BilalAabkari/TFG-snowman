import numpy as np
import gym
from gym import spaces
import math
import copy
import os
import random
from .SnowmanConstants import SnowmanConstants 

class SnowmanEnvironment(gym.Env):

    ENCODED_TEXT_MODE = 0
    DECODED_TEXT_MODE = 1
    GRAPHIC_MODE = 2 #futur, encara no implementat

    NO_PREPROCESS = 0
    PREPROCESS_V1 = 1
    PREPROCESS_V2 = 2
    PREPROCESS_V3 = 3

    STEP_BACK_PENALIZATION = 100
    BLOCKED_SNOWBALL_PENALIZATION = 100
    INCORRET_NUMBER_OF_SNOWBALLS_PENALIZATION = 100

    CLOSER_DISTANCE_BOUNUS = 10
    VISITED_PENALIZATION_MULTIPLIER = 1.2

    LESS_PUSHABLE_POSITIONS_PENALIZATION = 50
    

    def __init__(self, map_file, 
                 n,
                 m,
                 preprocess_mode,
                 enable_step_back_optimzation=False, 
                 enable_blocked_snowman_optimization=False, 
                 enable_snowball_number_optimization=False, 
                 enable_snowball_distances_optimization=False,
                 enable_visited_cells_optimization=False,
                 enable_pushable_positions_optimization=False):
        super(SnowmanEnvironment, self).__init__()
        self.n = n
        self.m = m 
        self.map, self.decoded_map = self._read_and_encode_map(map_file, n, m)
        self.original_map = copy.deepcopy(self.map) #Conserve the original map to allow reset    
        self.enable_snowball_number_optimization = enable_snowball_number_optimization
        self.enable_snowball_distances_optimization = enable_snowball_distances_optimization
        self.enable_visited_cells_optimization = enable_visited_cells_optimization
        self.previous_sum_of_distances = -100000
        self.preprocess_mode = preprocess_mode
        self.enable_pushable_positions_optimization = enable_pushable_positions_optimization
        self.previous_pushable_positions = None

        if self.preprocess_mode == self.NO_PREPROCESS:
            self.layers = 1
        elif self.preprocess_mode == self.PREPROCESS_V1:
            self.layers = 7
        elif self.preprocess_mode == self.PREPROCESS_V2:
            self.layers = 3
        elif self.preprocess_mode == self.PREPROCESS_V3:
            self.layers = 4
        self.visited = np.zeros((n,m))


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
        self.visited = np.zeros((self.n,self.m))
        self.agent_position = copy.deepcopy(self.original_agent_position)
        self.previous_agent_position = None
        self.previous_sum_of_distances = -100000

        return self.preprocess_map(self.map), { "Agent position" : self.agent_position}

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
                self.visited[a,b] = self.visited[a,b] + 1
                if self.enable_visited_cells_optimization:
                    reward = reward - self.visited[a,b]*self.VISITED_PENALIZATION_MULTIPLIER

                #print(self.agent_position)

        
            
        reward, critical_done = self.post_adjust_reward(reward)

        
        done = done or critical_done

        return self.preprocess_map(self.map),reward,done, {}  
    
    def move_agent_to_position(self, x, y):
        i, j = self.agent_position
        if i != x or j !=y:
            #print("moving...")
            if self.map[x,y] == SnowmanConstants.GRASS_CELL and self.map[i,j] == SnowmanConstants.CHARACTER_ON_GRASS_CELL:
                self.map[x,y] = SnowmanConstants.CHARACTER_ON_GRASS_CELL
                self.map[i,j] = SnowmanConstants.GRASS_CELL
            elif self.map[x,y] == SnowmanConstants.GRASS_CELL and self.map[i,j] == SnowmanConstants.CHARACTER_ON_SNOW_CELL:
                self.map[x,y] = SnowmanConstants.CHARACTER_ON_GRASS_CELL
                self.map[i,j] = SnowmanConstants.SNOW_CELL
            elif self.map[x,y] == SnowmanConstants.SNOW_CELL and self.map[i,j] == SnowmanConstants.CHARACTER_ON_GRASS_CELL:
                self.map[x,y] = SnowmanConstants.CHARACTER_ON_SNOW_CELL
                self.map[i,j] = SnowmanConstants.GRASS_CELL
            elif self.map[x,y] == SnowmanConstants.SNOW_CELL and self.map[i,j] == SnowmanConstants.CHARACTER_ON_SNOW_CELL:
                self.map[x,y] = SnowmanConstants.CHARACTER_ON_SNOW_CELL
                self.map[i,j] = SnowmanConstants.SNOW_CELL
            self.agent_position = (x,y)

    
    def get_valid_actions(self):
        valid_actions = []
        pushable_positions = self.generate_push_layer(self.map, 0, np.zeros((1, self.n, self.m)))
        accessibility_layer = self.generate_reachable_positions_layer(self.map, 0, np.zeros((1, self.n, self.m)))
        for i in range(self.n):
            for j in range(self.m):
                if accessibility_layer[0,i,j] == 1 and pushable_positions[0,i,j] == 1:
                    valid_movements, _ = self.get_valid_movements((i,j))
                    for k in valid_movements:
                        index = i*(self.m*4)+j*4+k
                        #print("valid movement: ", i,j,k, " (",index ,")")
                        valid_actions.append(index)


        return valid_actions
                
                
    
    def get_valid_movements(self, agent_position):
        x, y = agent_position
        actions = [(0,1),(1,0),(0,-1),(-1,0)]
        valid_actions=[]
        invalid_actions=[]
        action = 0
        for mov_x, mov_y in actions:
            next_pos = [x+mov_x, y+mov_y]
            next_of_next_pos = [next_pos[0]+mov_x, next_pos[1]+mov_y]
            dumb = (self.map[next_pos[0],next_pos[1]] == SnowmanConstants.SMALL_BALL_CELL and (self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SMALL_BALL_CELL or
                                                                                              self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SMALL_BALL_ON_LARGE_BALL_CELL or
                                                                                              self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SMALL_BALL_ON_MEDIUM_BALL_CELL or
                                                                                              self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.WALL_CELL)
                    or self.map[next_pos[0],next_pos[1]] == SnowmanConstants.MEDIUM_BALL_CELL and (self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.MEDIUM_BALL_CELL or 
                                                                                                   self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.MEDIUM_BALL_ON_LARGE_BALL_CELL or 
                                                                                                   self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SMALL_BALL_CELL or
                                                                                                   self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SMALL_BALL_ON_LARGE_BALL_CELL or
                                                                                                   self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SMALL_BALL_ON_MEDIUM_BALL_CELL or
                                                                                                   self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.WALL_CELL) 
                    or self.map[next_pos[0],next_pos[1]] == SnowmanConstants.LARGE_BALL_CELL and (self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.LARGE_BALL_CELL or 
                                                                                                  self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.MEDIUM_BALL_CELL or 
                                                                                                  self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SMALL_BALL_CELL or 
                                                                                                  self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.MEDIUM_BALL_ON_LARGE_BALL_CELL or
                                                                                                  self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SMALL_BALL_ON_MEDIUM_BALL_CELL or
                                                                                                  self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SMALL_BALL_ON_LARGE_BALL_CELL  or
                                                                                                  self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.WALL_CELL)
                    or self.map[next_pos[0],next_pos[1]] == SnowmanConstants.MEDIUM_BALL_ON_LARGE_BALL_CELL and not (self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.GRASS_CELL
                                                                                                  or self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SNOW_CELL)
                    or self.map[next_pos[0],next_pos[1]] == SnowmanConstants.SMALL_BALL_ON_LARGE_BALL_TOKEN and not (self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.GRASS_CELL
                                                                                                  or self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SNOW_CELL)
                    or self.map[next_pos[0],next_pos[1]] == SnowmanConstants.SMALL_BALL_ON_MEDIUM_BALL_CELL and not (self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.GRASS_CELL
                                                                                                  or self.map[next_of_next_pos[0],next_of_next_pos[1]] == SnowmanConstants.SNOW_CELL))

            can_not_push = (self.map[next_pos[0],next_pos[1]] == SnowmanConstants.GRASS_CELL or 
                            self.map[next_pos[0],next_pos[1]] == SnowmanConstants.SNOW_CELL or
                            self.map[next_pos[0],next_pos[1]] == SnowmanConstants.CHARACTER_ON_SNOW_CELL or
                            self.map[next_pos[0],next_pos[1]] == SnowmanConstants.CHARACTER_ON_GRASS_CELL)

            if dumb or self.map[next_pos[0],next_pos[1]] == SnowmanConstants.WALL_CELL or self.map[next_pos[0],next_pos[1]] == SnowmanConstants.OUT_OFF_GRID_CELL or can_not_push:
                invalid_actions.append(action)
            else:
                valid_actions.append(action)
            
            action = action + 1

        return valid_actions, invalid_actions
    
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
        return self.preprocess_map(self.map), {}

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
                                     self.map[next_cell[0],next_cell[1]] == SnowmanConstants.MEDIUM_BALL_CELL or 
                                     self.map[next_cell[0],next_cell[1]] == SnowmanConstants.LARGE_BALL_CELL)
            
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
            full_snowman = 0

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
                    elif self.map[i,j] == SnowmanConstants.FULL_SNOW_MAN_CELL:
                        full_snowman = full_snowman + 1
            
            large_balls = large_balls + medium_on_large_balls + small_on_large_balls + full_snowman
            medium_balls = medium_balls + medium_on_large_balls + small_on_medium_balls + full_snowman
            small_balls = small_balls + small_on_large_balls + small_on_medium_balls + full_snowman

            if ((large_balls == 0 and medium_balls == 0 and small_balls < 3) or
                (large_balls == 0 and (medium_balls+small_balls < 3 or small_balls == 0)) or 
                (large_balls > 0 and medium_balls == 0 and small_balls < 2) or
                (large_balls > 0 and medium_balls > 0 and small_balls == 0)):
                adjustedReward = adjustedReward - self.INCORRET_NUMBER_OF_SNOWBALLS_PENALIZATION
                critical_done = True

        if self.enable_snowball_distances_optimization:
            snowballs = []
            for i in range(self.n):
                for j in range(self.m):
                    if (self.map[i,j] == SnowmanConstants.SMALL_BALL_CELL or 
                        self.map[i,j] == SnowmanConstants.MEDIUM_BALL_CELL or 
                        self.map[i,j] == SnowmanConstants.LARGE_BALL_CELL or 
                        self.map[i,j] == SnowmanConstants.MEDIUM_BALL_ON_LARGE_BALL_CELL):
                        snowballs.append((i,j))

            if len(snowballs) >= 2:
                #print('snowballs', snowballs)
                sum_of_distances = 0
                for i in range(0, len(snowballs)-1):
                    snowball_reference_x,  snowball_reference_y = snowballs[i]
                    for j in range(i+1, len(snowballs)): 
                        x, y = snowballs[j]
                        sum_of_distances = sum_of_distances + math.sqrt(abs(snowball_reference_x-x)**2 + abs(snowball_reference_y-y)**2)
                        #print("distance between (" + str(snowball_reference_x) + "," + str(snowball_reference_y) + ") and (" + str(x) + "," + str(y) + ") is " + str(math.sqrt(abs(snowball_reference_x-x)**2 + abs(snowball_reference_y-y)**2)))                      


                if sum_of_distances < self.previous_sum_of_distances:
                    adjustedReward = adjustedReward + self.CLOSER_DISTANCE_BOUNUS
                    #print("bonus rewarded")
                elif sum_of_distances > self.previous_sum_of_distances:
                    adjustedReward = adjustedReward - self.CLOSER_DISTANCE_BOUNUS
                    
                self.previous_sum_of_distances = sum_of_distances
        if self.enable_pushable_positions_optimization:
            current_pushable_positions = len(self.get_pushable_positions(self.map))
            if self.previous_pushable_positions != None:
                if current_pushable_positions < self.previous_pushable_positions:
                    adjustedReward = adjustedReward - self.LESS_PUSHABLE_POSITIONS_PENALIZATION
            self.previous_pushable_positions = current_pushable_positions

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

    def preprocess_map(self, map):
        if self.preprocess_mode == self.NO_PREPROCESS:
            return [map]
        elif self.preprocess_mode == self.PREPROCESS_V1:
            return self.split_map_layers(map)
        elif self.preprocess_mode == self.PREPROCESS_V2:
            return self.split_map_layersV2(map)
        elif self.preprocess_mode == self.PREPROCESS_V3:
            return self.split_map_layersV3(map)


    def split_map_layers(self, state):
        transform=[[0],[4],[5],[4,5],[6],[4,6],[5,6],[4,5,6],[2],[1],[1,3],[2,3]]
        splitted_map=np.zeros((7, self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                for k in transform[int(state[i][j])]:
                    splitted_map[k,i,j]=1
        return splitted_map
    
    def split_map_layersV2(self, state):
        splitted_map=np.zeros((3, self.n, self.m)) #que posar si hi ha parets?
        for i in range(self.n):
            for j in range(self.m):
                splitted_map[0,i,j]=state[i,j]

        splitted_map = self.generate_reachable_positions_layer(state, 1, splitted_map)
        splitted_map = self.generate_push_layer(state, 2, splitted_map)

        return splitted_map
    
    def split_map_layersV3(self, state):
        splitted_map=np.zeros((4, self.n, self.m)) #que posar si hi ha parets?
        for i in range(self.n):
            for j in range(self.m):
                splitted_map[0,i,j]=state[i,j]

        splitted_map = self.generate_reachable_positions_layer(state, 1, splitted_map)
        splitted_map = self.generate_push_layer(state, 2, splitted_map)
        splitted_map = self.generate_visited_cells_layer(self.visited, 3, splitted_map)

        return splitted_map
    
    def generate_visited_cells_layer(self, visited, layer, result):
        splitted_map = result
        for i in range(self.n):
            for j in range(self.m):
                    splitted_map[layer, i, j] = visited[i,j]
        return splitted_map

    
    def generate_reachable_positions_layer(self, state, layer, result):
        splitted_map = self.generate_BFS_layer(state, layer, result)
        for i in range(self.n):
            for j in range(self.m):
                if splitted_map[layer, i, j] > 0:
                    splitted_map[layer, i, j] = 1
        return splitted_map
        

    def generate_BFS_layer(self, state, layer, result):
        splitted_map = result
        x,y = self.agent_position
        splitted_map[layer, x,y] = 1 
        moves = [(0,1),(1,0),(0,-1),(-1,0)]
        
        changes = True
        current_height = 0
        last_changes = []

        for mov_x, mov_y in moves:
            next_x = x+mov_x
            next_y = y+mov_y
            if state[next_x,next_y] == SnowmanConstants.GRASS_CELL or state[next_x,next_y] == SnowmanConstants.SNOW_CELL:
                splitted_map[layer, next_x, next_y] = current_height+1
                last_changes.append((next_x, next_y))

        current_height = current_height + 1

        while changes:
            changes = False
            new_changes = []
            for i, j in last_changes:
                for mov_i, mov_j in moves:
                    next_i = i+mov_i
                    next_j = j+mov_j #que posar a les parets i a l'agent
                    if splitted_map[layer, next_i, next_j] == 0 and (state[next_i,next_j] == SnowmanConstants.GRASS_CELL or state[next_i,next_j] == SnowmanConstants.SNOW_CELL):
                        splitted_map[layer, next_i, next_j] = current_height+1
                        new_changes.append((next_i, next_j))
                        changes = True
            last_changes = new_changes
            current_height = current_height+1

        return splitted_map

    def generate_push_layer(self, state, layer, result):
        splitted_map = result
        pushable_positions = self.get_pushable_positions(state)
        for i, j in pushable_positions:
            splitted_map[layer, i, j] = 1
        
        return splitted_map

    
    def get_pushable_positions(self, state):
        moves = [(0,1),(1,0),(0,-1),(-1,0)]
        pushable_positions = []
        for i in range(self.n):
            for j in range(self.m):
                if (state[i,j]== SnowmanConstants.GRASS_CELL or 
                    state[i,j]==SnowmanConstants.SNOW_CELL or 
                    state[i,j]==SnowmanConstants.CHARACTER_ON_GRASS_CELL or 
                    state[i,j]==SnowmanConstants.CHARACTER_ON_SNOW_CELL):
                    for mov_i, mov_j in moves:
                        next_i = mov_i + i
                        next_j = mov_j + j
                        next_of_next_i = mov_i + next_i
                        next_of_next_j = mov_j + next_j
                        

                        if (next_of_next_i > 0 and next_of_next_i < self.n and 
                            next_of_next_j > 0 and next_of_next_j < self.m and 
                            (state[next_i,next_j]==SnowmanConstants.SMALL_BALL_CELL or state[next_i,next_j]==SnowmanConstants.MEDIUM_BALL_CELL or state[next_i,next_j]==SnowmanConstants.LARGE_BALL_CELL)):
                            
                            can_push = (state[next_i,next_j]==SnowmanConstants.SMALL_BALL_CELL and (state[next_of_next_i,next_of_next_j]==SnowmanConstants.MEDIUM_BALL_CELL or
                                                                                                    state[next_of_next_i,next_of_next_j]==SnowmanConstants.LARGE_BALL_CELL or
                                                                                                    state[next_of_next_i,next_of_next_j]==SnowmanConstants.GRASS_CELL or 
                                                                                                    state[next_of_next_i,next_of_next_j]==SnowmanConstants.SNOW_CELL or
                                                                                                    state[next_of_next_i,next_of_next_j]==SnowmanConstants.CHARACTER_ON_GRASS_CELL or 
                                                                                                    state[next_of_next_i,next_of_next_j]==SnowmanConstants.CHARACTER_ON_SNOW_CELL)
                                        or state[next_i,next_j]==SnowmanConstants.MEDIUM_BALL_CELL and (state[next_of_next_i,next_of_next_j]==SnowmanConstants.LARGE_BALL_CELL or
                                                                                                        state[next_of_next_i,next_of_next_j]==SnowmanConstants.GRASS_CELL or 
                                                                                                        state[next_of_next_i,next_of_next_j]==SnowmanConstants.SNOW_CELL or
                                                                                                        state[next_of_next_i,next_of_next_j]==SnowmanConstants.CHARACTER_ON_GRASS_CELL or 
                                                                                                        state[next_of_next_i,next_of_next_j]==SnowmanConstants.CHARACTER_ON_SNOW_CELL)
                                        or state[next_i,next_j]==SnowmanConstants.LARGE_BALL_CELL and (state[next_of_next_i,next_of_next_j]==SnowmanConstants.GRASS_CELL or 
                                                                                                        state[next_of_next_i,next_of_next_j]==SnowmanConstants.SNOW_CELL or 
                                                                                                        state[next_of_next_i,next_of_next_j]==SnowmanConstants.CHARACTER_ON_GRASS_CELL or 
                                                                                                        state[next_of_next_i,next_of_next_j]==SnowmanConstants.CHARACTER_ON_SNOW_CELL))
                            if can_push:
                                pushable_positions.append((i,j))

        return pushable_positions