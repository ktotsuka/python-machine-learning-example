from ch18_grid_world_env import GridWorldEnv
from ch18_grid_world_agent import Agent
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

# Seed random generator for consistent results
np.random.seed(1)

# Create a new tuple type
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Function to run Q-learning
def run_qlearning(agent, env, num_episodes=50):
    history = []
    for episode in range(num_episodes):
        # Reset the grid world environment
        state = env.reset()
        # Display the environment
        env.render()
        # Reset the history parameters
        final_reward, n_moves = 0.0, 0

        while True:
            # Choose an action
            action = agent.choose_action(state)
            # Perform the action
            next_s, reward, done, _ = env.step(action)
            # Learn from the action
            agent._learn(Transition(state, action, reward, next_s, done))
            # Update the grid world display
            env.render(done=done)
            # Update parameters
            state = next_s
            n_moves += 1
            final_reward = reward
            # Break out of the loop if done
            if done:
                break
        # Update the history
        history.append((n_moves, final_reward))
        print('Episode %d: Reward %.1f #Moves %d'
              % (episode, final_reward, n_moves))

    return history

# Function to plot the learning history
def plot_learning_history(history):
    # create a figure
    fig = plt.figure(1, figsize=(14, 10))

    # Plot number of moves vs. episodes
    ax = fig.add_subplot(2, 1, 1)
    episodes = np.arange(len(history))
    moves = np.array([h[0] for h in history])
    plt.plot(episodes, moves, lw=4, marker="o", markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('# moves', size=20)

    # Plot the final reward (1: pass, 0: fail) vs. episodes
    ax = fig.add_subplot(2, 1, 2)
    rewards = np.array([h[1] for h in history])
    plt.step(episodes, rewards, lw=4)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('Final rewards', size=20)

    # Display the plots
    plt.show(block=True)


if __name__ == '__main__':
    # Create the grid world environment
    env = GridWorldEnv(num_rows=5, num_cols=6)

    # Instanciate the agent for the environment
    agent = Agent(env)

    # Run the Q-learning algorithm
    history = run_qlearning(agent, env, num_episodes=50)

    # Close the environment
    env.close()

    # Plot the learning history
    plot_learning_history(history)
