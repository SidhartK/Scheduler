import gymnasium as gym

from stable_baselines3 import SAC


env = gym.make("Pendulum-v1")
import pdb; pdb.set_trace()

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

env = gym.make("Pendulum-v1", render_mode='human')
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()