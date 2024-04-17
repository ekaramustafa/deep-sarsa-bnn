import numpy as np
import gymnasium as gym
from utils.enviroment_wrapper import EnvironmentWrapper
from layers.adin_vb import AdinVBL, AdinVBConv2d
from agents.dql import DQLAgent
from agents.des import DESAgent

import torch 


def make_env():


    pass

def main():
    seed = 42
    env = gym.make("BreakoutNoFrameskip-v4")
    env = EnvironmentWrapper.wrap_environment(env)
    agent = DESAgent(env,is_deterministic=False,linear_layer_class=AdinVBL,conv_layer_class=AdinVBConv2d)
    agent.train(num_episodes=1)


    pass



if __name__ == '__main__':
    main()