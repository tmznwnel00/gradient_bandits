import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.config.run_functions_eagerly(True)

# states = random states generated for training, 
# total_states = possible states count
def ohe_generator(states, total_states):
    ohe = np.zeros((len(states), total_states))
    for index, array in enumerate(ohe):
        ohe[index][states[index]] = 1
    return ohe

class contextual_bandits:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
    
    def reward(self, state, action):
        if (state * action) % 2 == 1:
            return 0.5 + 0.05 * ((state + action) % 10) + np.random.rand() * 0.1
        else:
            return 0.9 - 0.1 * ((state + action) % 10) + np.random.rand() * 0.1
    
    def network(self):
        input_ = Input(shape=(self.states,))
        dense1 = Dense(128, activation='relu')(input_)
        dropout1 = Dropout(0.1)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.1)(dense2)
        dense3 = Dense(self.actions, activation='sigmoid')(dropout2)
        model = Model(input_, dense3)
        
        rms = Adam(learning_rate=0.0001)
        model.compile(loss="mean_absolute_error", optimizer=rms, metrics=["mean_absolute_error"])
        return model   
    
batch_size = 128
states = 100
actions = 4

def training():
    cb = contextual_bandits(states, actions)
    model = cb.network()
    sample_states = np.random.choice(range(states), size=batch_size*100)
    state_ohe = ohe_generator(sample_states, states)
    actual_reward = [[cb.reward(x, y) for y in range(cb.actions)] for x in sample_states]
    actual_reward_matrix = np.zeros((len(state_ohe), cb.actions))
    for index, x in enumerate(actual_reward):
        actual_reward_matrix[index] = np.array(x)
    model.fit(state_ohe, actual_reward_matrix, batch_size=batch_size, epochs=20) 
    return model

# Train the model and save it for later use
model = training()

# Now, use the trained model to predict the estimated rewards
state_ohe = ohe_generator(np.array([x for x in range(100)]), states)
estimated_reward = model.predict(state_ohe)

# Print the best action for each state based on the estimated rewards
print({x: np.argmax(y) for x, y in enumerate(estimated_reward)})

# Testing the reward function for state 0 and 93
cb = contextual_bandits(100, 4)
print('\nReward for state {}\n'.format(0))
for x in range(4):
    print(cb.reward(0, x))
    
print('\nReward for state {}\n'.format(93))
for x in range(4):
    print(cb.reward(93, x))
