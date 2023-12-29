import gymnasium as gym

env = gym.make("voxelgym2D:onestep-v0")
observation, info = env.reset(seed=123456)

done = False
while not done:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated
    env.render()

env.close()
