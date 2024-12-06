from .training_env import DAGEnv

def sample_data(env, num_samples=1000):
    data = []
    for _ in range(num_samples):
        action = env.action_space.sample()
        state, _, done, _ = env.step(action)
        data.append(state[-len(env.dag.nodes) * 2:])
        if done:
            env.reset()
    return data

def fit(data):
    