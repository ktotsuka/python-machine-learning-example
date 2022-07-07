from collections import defaultdict
import numpy as np

# Agent class that will interract with the grid world environment
class Agent(object):
    def __init__(
            self, env,
            learning_rate=0.01, # higher the value, the more adjustment is made for the predicted value for a state (p692)
            discount_factor=0.9, # higher the value, the future reward worth more (see p681)
            epsilon_greedy=0.9, # higher the value, the more likely to take random actions
            epsilon_min=0.1,
            epsilon_decay=0.95):
        # Set up parameters
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Define the q_table, q[state][action] (=action-value function, see p683) 
        self.q_table = defaultdict(lambda: np.zeros(self.env.nA))

    # Function to choose an action in the grid world environment
    def choose_action(self, state):
        # Check if a randomly generated number is less than the epsilon
        # If so, choose the action randomly
        # Otherwise, use choose the action with the best estimated return
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.env.nA)
        else:
            q_vals = self.q_table[state]
            perm_actions = np.random.permutation(self.env.nA) # mix the order so that random action will be taken if estimated returns are the same
            q_vals = [q_vals[a] for a in perm_actions]
            perm_q_argmax = np.argmax(q_vals)
            action = perm_actions[perm_q_argmax]
        return action

    def _learn(self, transition):
        # Unpack the tuple
        s, a, r, next_s, done = transition # state, action, reward, next state, done?
        q_current_estimated_return = self.q_table[s][a]
        # Update the estimated return based on the Q-learning algorithm (see p694)
        if done:
            q_target_return = r
        else:
            q_target_return = r + self.gamma*np.max(self.q_table[next_s])
        self.q_table[s][a] += self.lr * (q_target_return - q_current_estimated_return)

        # Adjust the epislon
        self._adjust_epsilon()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
