import numpy as np
import pygame
import random
#import gym
from rocket_gym import RocketMeister10
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


from tensorflow.keras import backend as K
from collections import deque

class DQNAgent():
    def __init__(self, states, actions, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay,max_memory=100000):
        self.nStates  = states
        self.nActions = actions
        self.memory = deque([], maxlen=max_memory)
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
        model.add(Dense(64, activation='relu'))    # Input + Layer 1
        model.add(Dense(32, activation='relu'))                            # Layer 2
        model.add(Dense(64, activation='relu'))                            # Layer 3
        model.add(Dense(self.nActions, activation='linear'))               # Layer 4 [output]

        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                      optimizer=tensorflow.keras.optimizers.Adam(learning_rate=self.learning_rate)) #Optimaizer: Adam (Feel free to check other options)
        return model

    def action(self, state):
        #Exploration
        if np.random.rand() <= self.epsilon: # Randomly take random actions
            rand_vec = np.random.rand(self.nActions)
            result = (np.abs(rand_vec)).argsort()

        # Exploitation
        else:
            action_vals =np.array(self.model.predict(state)[0]) # Use the NN to predict the correct action from this state
            result = (np.abs(action_vals)).argsort()

        return self.define_action_from_vect(result)

    def define_action_from_vect(self, actions):
        # Possible actions : Acce - Rec - G - D - Nothing
        # Acc : [1,0]
        # Rec : [-1,0]
        # G : [0,1]
        # D : [0,-1]
        # Nothing : [0,0]
        # Cumul : Acc/Gauche [1,1] - Acc/Droite [1,-1] - Rec/Gauche [-1,1] - Rec/Droite [-1,-1]
        if np.argmax(actions)==0:
            return [1,0]
        if np.argmax(actions)==1:
            return [-1,0]
        if np.argmax(actions)==2:
            return [0,1]
        if np.argmax(actions)==3:
            return [0,-1]
        if np.argmax(actions)==4:
            return [0,0]
        if np.argmax(actions)==5:
            return [1,1]
        if np.argmax(actions)==6:
            return [1,-1]
        if np.argmax(actions)==7:
            return [-1,1]
        if np.argmax(actions)==8:
            return [-1,-1]


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
        
        st = np.zeros((0,self.nStates)) #States
        nst = np.zeros((0,self.nStates) )#Next States




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
        self.memory = deque([], maxlen=max_memory)

    def load(self, name):
        self.model.load_weights(name)
        #print("Model Loaded.")

    def save(self, name):
        self.model.save_weights(name)
        #print("Model Saved.")


agent_max_memory = 1000

agent = DQNAgent(states         = 10, 
                 actions        = 9,
                 learning_rate  = 0.005, 
                 gamma          = 0.99, 
                 epsilon        = 1, 
                 epsilon_min    = 0.001, 
                 epsilon_decay  = 0.995,
                 max_memory=agent_max_memory) 



# ─── INITIALIZE AND RUN ENVIRONMENT ─────────────────────────────────────────────
env_config = {
    'gui': True,
    #'env_name': 'default',
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
    'gui_draw_goal_points': True,
}







def start_agent_traning(view: bool,agentName: str):
    try:
        agent.load(agentName)
    except:
        print("Unable to load agent")


    env = RocketMeister10(env_config)
    state = np.reshape(env.reset(),[1,10])
    
    while True:
        try:
            action = agent.action(state)

            
            if view : 
                env.render()
                env.clock.tick(120)
                

            nstate, reward, done, info = env.step(action)
            nstate = np.reshape(nstate,[1,10])
            
            agent.memorize(state,action,reward,nstate,done)
            
            state = nstate

            if done:
                print(f"La mémoire de l'agent est complétée à : {(len(agent.memory)/agent_max_memory)*100}%")
                # Remember program logic
                env.reset()
                agent.experience_replay(100)
                try:
                    agent.save("hugodemenez")
                except:
                    print("Unable to save agent")
                    

        except Exception as error:
            print(f'Game exit because of error : {error}')
            if view :
                env.close()
                pygame.quit()
            break
    



if __name__=='__main__':
    start_agent_traning(True,"hugodemenez")