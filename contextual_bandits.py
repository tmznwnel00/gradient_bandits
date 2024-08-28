import tensorflow as tf
import numpy as np

class contextual_bandit():
    def __init__(self):
        self.state = 0
        # List out our bandits. 
        self.bandits = np.array([
            [0.2, 0, 0 ,-5],          # Currently arms 4
            [0.1, -5, 1, 0.25],       # 2 and 
            [-5, 5, 5, 5]             # 1 are the most optimal.(respectively)
        ])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]
        
    def getBandit(self):
        self.state = np.random.randint(0, len(self.bandits))  # Get a random state for each episode.
        return self.state
        
    def pullArm(self, action):
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            # Return a positive reward.
            return 1
        else:
            # Return a negative reward.
            return -1

class agent():
    def __init__(self, lr, s_size, a_size):
        # Define the input for the state.
        self.state_in = tf.keras.Input(shape=(1,), dtype=tf.int32)
        state_in_OH = tf.one_hot(self.state_in, s_size)

        # Define the fully connected layer using tf.keras.layers
        output = tf.keras.layers.Dense(a_size, activation='sigmoid', kernel_initializer=tf.ones_initializer())(state_in_OH)
        
        self.model = tf.keras.Model(inputs=self.state_in, outputs=output)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    def choose_action(self, state):
        logits = self.model(np.array([state]))
        return np.argmax(logits)

    def update(self, state, action, reward):
        with tf.GradientTape() as tape:
            logits = self.model(np.array([state]))
            responsible_weight = tf.gather(logits, action, axis=1)
            loss = -tf.math.log(responsible_weight) * reward
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

cBandit = contextual_bandit()  # Load the bandits.
myAgent = agent(lr=0.001, s_size=cBandit.num_bandits, a_size=cBandit.num_actions)  # Load the agent.

total_episodes = 10000  # Set total number of episodes to train agent on.
total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])  # Set scoreboard for bandits to 0.
e = 0.1  # Set the chance of taking a random action.

# Train the agent
for i in range(total_episodes):
    s = cBandit.getBandit()  # Get a state from the environment.
    
    # Choose either a random action or one from our network.
    if np.random.rand(1) < e:
        action = np.random.randint(cBandit.num_actions)
    else:
        action = myAgent.choose_action(s)
    
    reward = cBandit.pullArm(action)  # Get our reward for taking an action given a bandit.
    
    # Update the network.
    myAgent.update(s, action, reward)
    
    # Update our running tally of scores.
    total_reward[s, action] += reward
    if i % 500 == 0:
        print("Mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " + str(np.mean(total_reward, axis=1)))

for a in range(cBandit.num_bandits):
    best_action = np.argmax(total_reward[a])
    print(f"The agent thinks action {best_action + 1} for bandit {a + 1} is the most promising....")
    if best_action == np.argmin(cBandit.bandits[a]):
        print("...and it was right!")
    else:
        print("...and it was wrong!")
