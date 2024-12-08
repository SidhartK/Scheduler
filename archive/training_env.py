from tqdm import trange
from generator import generate_dag, assign_errors
import gym
from collections import deque
import numpy as np
import networkx as nx
import pandas as pd

class DAGEnv(gym.Env):
    def __init__(self, dag, history_len=10, time_limit=100):
        super(DAGEnv, self).__init__()
        self.dag = dag
        self.history_len = history_len
        self.time_limit = time_limit
        self.action_space = gym.spaces.Box(low=-20.0, high=20.0, shape=(len(self.dag.nodes),), dtype=float)
        self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=(len(self.dag.nodes) * 2 * history_len,), dtype=float)
        self.nodes = list(nx.topological_sort(self.dag))

    def reset(self):
        self.history = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.history.append((np.zeros(len(self.dag.nodes)), np.zeros(len(self.dag.nodes))))
        self.best_error = None
        self.t = 0
        return self._get_state()

    def step(self, action):
        compute_distribution = np.exp(action - np.max(action)) / np.sum(np.exp(action - np.max(action)))

        for i, node in enumerate(self.dag.nodes):
            self.dag.nodes[node]["compute"] = compute_distribution[i]

        error = assign_errors(self.dag)

        if self.best_error is None:
            reward = -error[self.nodes[-1]]
            self.best_error = error[self.nodes[-1]]
        else:
            reward = self.best_error - min(self.best_error, error[self.nodes[-1]])
            self.best_error = min(self.best_error, error[self.nodes[-1]])
        

        self.history.append((action, error))
        done = self.t >= self.time_limit
        self.t += 1
        return self._get_state(), reward, done, {}

    def _get_state(self):
        state = []
        for action, errors in self.history:
            state.extend(action)
            state.extend(errors)
        return state

    def render(self, mode='human'):
        pass

def random_solution(env, num_episodes=1000):
    rewards = []
    for _ in trange(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return avg_reward, std_reward

if __name__ == "__main__":
    avg_rewards = []
    std_rewards = []
    edge_prob = 0.2
    for num_nodes in range(5, 15):
        # for edge_prob in np.linspace(0.1, 0.9, 9):
        print("-"*50)
        print(f"Number of Nodes: {num_nodes}, Edge Probability: {edge_prob}")
        dag = generate_dag(num_nodes, edge_prob)
        for u, v in dag.edges:
            dag[u][v]['weight'] = np.random.uniform(0.1, 1)
        env = DAGEnv(dag, time_limit=1)
        avg_reward, std_reward = random_solution(env)
        print(f"Average Reward: {avg_reward}, Standard Deviation: {std_reward}")
        avg_rewards.append(avg_reward)
        std_rewards.append(std_reward)
