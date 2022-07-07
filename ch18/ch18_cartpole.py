import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from collections import deque

# Seed random generators for consistent results
np.random.seed(1)
tf.random.set_seed(1)

# Create a new tuple type
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Agent that will interract with the cartpole environment
class DQNAgent:
    def __init__(
            self,
            env,
            discount_factor=0.95, # higher the value, the future reward worth more (see p681)
            epsilon_greedy=1.0, # higher the value, the more likely to take random actions
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=1e-3, # higher the value, the more adjustment is made for the predicted value for a state (p692)
            max_memory_size=2000):
        # Set up parameters
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate

        # Initialize the replay memory (p710)
        self.memory = deque(maxlen=max_memory_size)

        # Build the neural network model
        self._build_nn_model()

    # Function to build a NN model
    # Input: state (cart position, cart velocity, pole angle, pole angular velocity)
    # Output: Estimated return for each action (left, right)
    def _build_nn_model(self, n_layers=3):
        # Create a bare model
        self.model = tf.keras.Sequential()

        # Add hidden layers
        for n in range(n_layers - 1):
            self.model.add(tf.keras.layers.Dense(units=32, activation='relu'))
            self.model.add(tf.keras.layers.Dense(units=32, activation='relu'))

        # Add the last layer
        self.model.add(tf.keras.layers.Dense(units=self.action_size))

        # Build & compile model
        self.model.build(input_shape=(None, self.state_size))
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.lr))

    def remember(self, transition):
        self.memory.append(transition)

    def choose_action(self, state):
        # If the random value is less than the epsilon, choose the action randomly
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # If not, predict the return for the current state for each action and choose the action with the higher return
        q_values = self.model.predict(state)[0] # Need [0] to change the shape from (1,2) to (2,)
        return np.argmax(q_values)  # returns action

    def _learn(self, batch_samples):
        # Initialize variables
        batch_states, batch_targets = [], []
        # For each transition samples, keep track of the state (input) and the target return (output)
        for transition in batch_samples:
            # Unpack the tuple
            s, a, r, next_s, done = transition
            # Update the estimated return based on the deep Q-learning algorithm (see p694 and p712)
            if done:
                target_return = r
            else:
                # The return estimate contains two values.  One for each action (left and right)
                # Need [0] to change the shape from (1,2) to (2,)
                return_estimate_for_next_state = self.model.predict(next_s)[0] 
                target_return = (r + self.gamma * np.amax(return_estimate_for_next_state))
            return_estimate_for_current_state = self.model.predict(s)[0] # this is the output of the current model as is
            return_estimate_for_current_state[a] = target_return # we want the model to change so that it outputs this 
            # Keep track of the input (state) and the output (return estimate)
            batch_states.append(s.flatten())
            batch_targets.append(return_estimate_for_current_state)
            # Adjust the epislon
            self._adjust_epsilon()
        # Train the model using the batch samples
        return self.model.fit(x=np.array(batch_states),
                              y=np.array(batch_targets),
                              epochs=1,
                              verbose=0)

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Function to train the model using some samples from the replay memory
    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        history = self._learn(samples)
        return history.history['loss'][0]

# Function to plot the learning history
def plot_learning_history(history):
    # create a figure
    fig = plt.figure(1, figsize=(14, 5))
    # Plot the total reward vs. episodes
    ax = fig.add_subplot(1, 1, 1)
    episodes = np.arange(len(history[0])) + 1
    plt.plot(episodes, history[0], lw=4, marker='o', markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('# Total Rewards', size=20)
    plt.show(block=True)


# General settings
EPISODES = 100
batch_size = 32
init_replay_memory_size = 500

if __name__ == '__main__':
    # Create the cart pole environment
    env = gym.make('CartPole-v1')

    # Print out information about the environment
    # Observation space: (4,): cart position, cart velocity, pole angle, pole angular velocity
    # Action space: 2: Push left or push right 
    print('observation space: ', env.observation_space)
    print('action space: ', env.action_space)

    # Instanciate the agent for the environment
    agent = DQNAgent(env)

    # Reset the cart pole environment
    state = env.reset()
    state = np.reshape(state, [1, agent.state_size])

    # Fill up the replay-memory
    for i in range(init_replay_memory_size):
        # Choose an action
        action = agent.choose_action(state)
        # Perform the action
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        # Enqueue this transition to the replay memory
        agent.remember(Transition(state, action, reward, next_state, done))
        # Update the state
        if done:
            state = env.reset()
            state = np.reshape(state, [1, agent.state_size])
        else:
            state = next_state

    # Run deep Q-Learning
    total_rewards = []
    for e in range(EPISODES): # this step takes very long time (Several hours)
        # Reset the environment
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        # Perform actions until done
        i = 0
        while True:
            # Choose an action
            action = agent.choose_action(state)
            # Perform the action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            # Enqueue this transition to the replay memory
            agent.remember(Transition(state, action, reward, next_state, done))
            # Update the state
            state = next_state
            # Update the display every 10 episodes
            if e % 10 == 0:
                env.render()
            # If done, capture the total reward for this episode
            if done:
                total_rewards.append(i)
                print('Episode: %d/%d, Total reward: %d' % (e, EPISODES, i))
                break
            # Train the model using the transitions in the replay memory
            agent.replay(batch_size)
            # Update counter
            i += 1

    # Plot the learning history
    plot_learning_history(total_rewards)
