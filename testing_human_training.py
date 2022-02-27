import numpy as np
import pygame
from rocket_gym import RocketMeister10

import tensorflow
from tensorflow import keras
from collections import deque
import random
# ─── FUNCTIONS FOR USER INPUT ───────────────────────────────────────────────────
def event_to_action(eventlist):
    global run
    for event in eventlist:
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            env.reset()

def pressed_to_action(keytouple):
    action_turn = 0.
    action_acc = 0.
    if keytouple[81] == 1:  # back
        action_acc -= 1
    if keytouple[82] == 1:  # forward
        action_acc += 1
    if keytouple[80] == 1:  # left  is -1
        action_turn += 1
    if keytouple[79] == 1:  # right is +1
        action_turn -= 1
    if keytouple[21] == 1:  # r, reset
        # game.reset()
        pass
    # ─── KEY IDS ─────────
    # arrow backwards : 81
    # arrow forward   : 82
    # arrow left      : 80
    # arrow right     : 79
    # r               : 21
    return np.array([action_acc, action_turn])
   
class DeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon,epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.loss = []
        
    def build_model(self):
        #model = tensorflow.keras.models.load_model("fusee.h5",compile = False) #Pour charger un modèle existant
        #model.compile(loss='categorical_crossentropy',  optimizer=keras.optimizers.Adam(lr=self.alpha)) #Loss function
        #return model
        model = keras.Sequential() #linear stack of layers https://keras.io/models/sequential/
        model.add(keras.layers.Dense(24, input_dim=self.nS, activation='tanh')) #[Input] -> Layer 1
        model.add(keras.layers.Dense(16, input_dim=self.nS, activation='tanh'))
        model.add(keras.layers.Dense(12, activation='tanh'))
        model.add(keras.layers.Dense(self.nA, activation='softmax'))
        #   Size has to match the output (different actions)
        model.compile(loss='categorical_crossentropy', #Loss function
        #   Linear activation on the last layer
                      optimizer=keras.optimizers.Adam(lr=self.alpha)) #Optimaizer: Adam 
        return model

    def action(self, state):
        
        if np.random.rand() <= self.epsilon:
            #return random.randrange(self.nA) #Explore
            rand_vec = np.random.rand(self.nA)
            acc , rot = np.split(rand_vec,2)
            rotation = np.argmax(rot)-1
            acceleration = np.argmax(acc)-1
            return [acceleration,rotation]
            
        action_vals =np.array( self.model.predict(state)[0] )#Exploit: Use the NN to predict the correct action from this state
        #result = (-action_vals).argsort()[:2]
        acc , rot = np.split(action_vals,2)
        rotation = np.argmax(rot)-1
        acceleration = np.argmax(acc)-1
        return [rotation,acceleration]
        #return np.argmax(action_vals[0])

    def test_action(self, state): #Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done) )

    def experience_replay(self, batch_size):
        #Execute the experience replay
        minibatch = random.sample( self.memory, batch_size() ) #Randomly sample from memory

        #Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch, dtype = object)
        st = np.zeros((0,self.nS)) #States
        nst = np.zeros( (0,self.nS) )#Next States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np_array[i,0], axis=0)
            nst = np.append( nst, np_array[i,3], axis=0)
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        index = 0
        for state, action, reward, nstate, done in minibatch:
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
            index += 1
        #Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size(),self.nS)
        y_reshape = np.array(y)
        epoch_count = 1 #Epochs is the number or iterations
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#Hyper Parameters
def discount_rate(): #Gamma
    return 0.95

def learning_rate(): #Alpha
    return 0.001

def batch_size(): #Size of the batch used in the experience replay
    return 48
    
    

# ─── INITIALIZE AND RUN ENVIRONMENT ─────────────────────────────────────────────
env_config = {
    'gui': True,
    'env_name': 'default',
    # 'env_name': 'empty',
     'env_name': 'level1',
    #'env_name': 'level2',
    #'env_name': 'random',
    #'camera_mode': 'centered',
    # 'env_flipped': False,
    # 'env_flipmode': False,
    # 'export_frames': True,
    'export_states': False,
    # 'export_highscore': False,
    'export_string': 'human',
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
nS = 10
nA=6
dqn = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.001, 0.995 )
env = RocketMeister10(env_config)
env.render()
run = True
reward_total = 0
rewards = [] #Store rewards for graphing
epsilons = [] # Store the Explore/Exploit
done=False
actionNN = [0,0]
state=env.reset()
state=np.reshape(state,[1,nS])
total_reward = 0    
while run:
    events = pygame.event.get() #appuyer sur s pour sauvegarder le modèle
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                dqn.model.save("fusee.h5")
            if event.key == pygame.K_ESCAPE:
                run = False
    env.clock.tick(120)
    get_event = pygame.event.get()
    event_to_action(get_event)
    #get_pressed = list(pygame.key.get_pressed()) #utiliser le clavier
    action = dqn.action(state)#le réseau de neuronne bouge la voiture
    #action = [1,1]
    nstate,reward,done, info = env.step(action=action) #résultat
    nstate = np.reshape(nstate, [1, nS])#reshape pour pouvoir stocker
    state = nstate #l'état devient le nouvel état
    env.render()
    total_reward = env.rocket.reward_total
    action_memoire = [0,0,0,0,0,0]
    action_memoire[action[0]+1]=1
    action_memoire[action[1]+4]=1
    #on crée une liste pour le réseau de neuronne, les 3 premiers sont pour l'accélération les 3 dernier pour la rotation
    #100 ralenti, 010 on ne fait rien, 001 accélère, pour la rotation 100 et 001 rotation 010 on ne fait rien



    dqn.store(state,action_memoire, reward, nstate, done)
    if done : #Si on meurt
        rewards.append(total_reward)
        epsilons.append(dqn.epsilon)
        env.reset()

    if len(dqn.memory) > batch_size():
        dqn.experience_replay(batch_size)
        
pygame.quit()
