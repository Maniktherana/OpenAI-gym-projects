import gym

env = gym.make('CartPole-v1')

observation = env.reset()

for t in range(1000):
    env.render()

    position, velocity, pole_angle, angular_velocity = observation
    # basic policy
    if pole_angle > 0:  # turn right if pole goes left
        action = 1
    else:
        action = 0

    observation, reward, done, info = env.step(action)
