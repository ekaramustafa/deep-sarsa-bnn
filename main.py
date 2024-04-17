import numpy as np
import gymnasium as gym
from utils.enviroment_wrapper import EnvironmentWrapper
from agents.dql import DQLAgent


import torch 


def make_env():


    pass

def main():
    seed = 42
    env = gym.make("BreakoutNoFrameskip-v4")
    env = EnvironmentWrapper.wrap_environment(env)
    agent = DQLAgent(env,is_deterministic=True)
    agent.train(num_episodes=2)


    pass



if __name__ == '__main__':
    main()