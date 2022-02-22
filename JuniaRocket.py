import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
import random
import numpy as np
import pygame
#import gym
import rocket_gym
from collections import deque

class DQNAgent():
    def __init__(self, states, actions, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay):
        self.nStates  = states
        self.nActions = actions
        self.memory = deque([], maxlen=500)
        self.learning_rate = learning_rate
        self.gamma = gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.loss = []
        
    def build_model(self):
        model = Sequential() 
        model.add(Dense(64, input_dim=self.nStates, activation='relu'))    # Input + Layer 1
        model.add(Dense(64, activation='relu'))                            # Layer 2
        model.add(Dense(64, activation='relu'))                            # Layer 3
        model.add(Dense(self.nActions, activation='linear'))               # Layer 4 [output]

        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                      optimizer=tensorflow.keras.optimizers.Adam(lr=self.learning_rate)) #Optimaizer: Adam (Feel free to check other options)
        return model

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nActions)    #Exploration
        action_vals = self.model.predict(state)       #Exploitation
        return np.argmax(action_vals[0])

    def bestAction(self, state):
        action_vals = self.model.predict(state) #Exploitation
        return np.argmax(action_vals[0])

    def memorize(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done) )

    def experience_replay(self, batch_size):
        if batch_size > len(self.memory) :
            return
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory

        #Convert to numpy for speed by vectorization
        x = []
        y = []
        #np_array = np.array(minibatch)
        #st = np.zeros((0,self.nS)) #States
        #nst = np.zeros((0,self.nS) )#Next States

        st = np.array([],dtype=int)
        nst = np.array([],dtype=int)  


        for state, action, reward, next_state, done in minibatch:
        #for i in range(len(np_array)): #Creating the state and next state np arrays
            #print(state, action + 1, reward, next_state, done)
            st = np.append( st, state, axis=0)
            nst = np.append( nst, next_state, axis=0)
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        for index, (state, action, reward, nstate, done) in enumerate(minibatch):
            x.append(state)
            #Predict from state
            nst_action_predict_model = nst_predict[index]
            if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * np.amax(nst_action_predict_model)
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
        #Reshape for tensorflow.keras Fit
        x_reshape = np.array(x).reshape(batch_size,self.nStates)
        y_reshape = np.array(y)
        epoch_count = 1 #Epochs is the number or iterations
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        #print("Model Loaded.")

    def save(self, name):
        self.model.save_weights(name)
        #print("Model Saved.")

################################################################

env_config = {
						'gui': True,
						# 'env_name': 'default',
						# 'env_name': 'empty',
						'env_name': 'level1',
						# 'env_name': 'level2',
						# 'env_name': 'random',
						# 'camera_mode': 'centered',
						# 'env_flipped': False,
						# 'env_flipmode': False,
						# 'export_frames': True,
						'export_states': False,
						# 'export_highscore': False,
						# 'export_string': 'human',
						'max_steps': 1000,
						'gui_reward_total': True,
						'gui_echo_distances': True,
						'gui_level': True,
						'gui_velocity': True,
						'gui_goal_ang': True,
						'gui_frames_remaining': True,
						'gui_draw_echo_points': True,
						'gui_draw_echo_vectors': True,
						'gui_draw_goal_points': True,}

env = rocket_gym.RocketMeister10(env_config)
observation = env.reset()
env.render()


agent = DQNAgent(states         = 10,
                 actions        = 2,
                 learning_rate  = 0.005,
                 gamma          = 0.99,
                 epsilon        = 0.05,
                 epsilon_min    = 0.05,
                 epsilon_decay  = 0.9995)



print(observation.shape)


for _ in range(10000):
  
  #action = [action_acc, action_turn]
  
  action = agent.action(observation) # predicting action with nimDQN agent
  
  # Doing actions and memorizing results
  observation, reward, done, info = env.step(action)
  agent.memorize(observation,action,reward,info,done)
  if done:
    observation = env.reset()
  env.render()
      

env.close()
pygame.quit()