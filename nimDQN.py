import numpy as np
import random

NB_STICKS = 24

################################################################

def playGame():
    sticks, player = NB_STICKS, 0
    while sticks > 0:
        choice = random.randint(1, min(sticks,3))
        print ("Player ", player, ":", sticks, "-", choice, "->", sticks-choice)
        sticks -= choice
        player = 1 - player
    print ("Player ", player, "won the game")

#print ("Generate Random Game")
#playGame();
#input("Press Enter to continue...")

################################################################

def playPredictedGame():
    sticks, player = NB_STICKS, 0
    while sticks > 0:
        choice = agent.action(np.reshape(sticks, (-1,1))) + 1
        # Adjust the action if not possible
        choice = min(sticks, choice)

        print ("Player ", player, ":", sticks, "-", choice, "->", sticks-choice)
        sticks -= choice
        player = 1 - player
    print ("Player ", player, "won the game")

################################################################
def generateGame(printStates=False):
    sticks, player = NB_STICKS, random.randint(0,1)
    states = np.array([],dtype=int)
    choices = np.array([],dtype=int)

    while sticks > 0:
        if player == 0:
            choice = agent.action(np.reshape(sticks, (-1,1))) + 1
            choice = min(sticks, choice) # for cases 1 and 2 stcks, to prevent from choosing 3

            # Append thecurrent state to the current game memory
            states = np.append (states, sticks)              
            # In the neural network :
            # Output neuron 0 -> 1 stick
            # Output neuron 1 -> 2 sticks
            # Output neuron 2 -> 3 sticks
            # Append the current action to the current game's action memory
            choices = np.append (choices, choice-1)

        else :
            choice = agent.bestAction(np.reshape(sticks, (-1,1))) + 1
            choice = min(sticks, choice) # for cases 1 and 2 stcks, to prevent from choosing 3

        sticks -= choice
        player = 1 - player

    #print ("Player ", player, "won the game // Reward for player 0")

    # Once the game ended, we can estimate the reward for each actions
    # to create the data and add it to memory
    reward = 1 if player == 0 else -1
    for i in (reversed(range(len(states)))):

        state = states[i]
        action = choices[i]      
        next_state = states[i+1] if i+1 < len(states) else 0

        state     = np.reshape(state, [1])
        next_state = np.reshape(next_state, [1])

        if printStates:
            print(state, action + 1, reward, next_state, next_state[0] == 0)
        agent.memorize(state, action, reward, next_state, next_state[0] == 0)

        #reward = - reward   # switch between winner and loser (1 and -1)

################################################################



import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


from tensorflow.keras import backend as K
from collections import deque

################################################################

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


agent = DQNAgent(states         = 1, 
                 actions        = 3, 
                 learning_rate  = 0.005, 
                 gamma          = 0.99, 
                 epsilon        = 0.05, 
                 epsilon_min    = 0.05, 
                 epsilon_decay  = 0.9995) 

################################################################


try :
    agent.load("hugodemenez.ai")
except: 
    pass

for i in range(10000):
    try:
        
        #print ("GAME :", i, "Epsilon = ", agent.epsilon)
        generateGame()
        agent.experience_replay(40)
        
        if i % 10 == 0:
            generateGame(printStates=True)
            for j in range(2,NB_STICKS+1):
                a = agent.model.predict([j])
                print(j, "->", np.argmax(a)+1, a)
            print("\n-------", i, agent.epsilon)
            
    # Save even in case of keyboard interrupt
    except KeyboardInterrupt:
        print("Saving progress....")

    finally:
        agent.save("hugodemenez.ai")




