import numpy as np
from networks.bnn import DQN_VBLinear
from layers.alter_vb import AlterVBL, AlterVBConv2d
from layers.adin_vb import AdinVBL, AdinVBConv2d
from test import FunctionApproxTester
from networks.bnn import DQN_VBLinear, DQN_VBConv2D
from gymnasium.wrappers import AtariPreprocessing, FrameStack,RecordVideo
import gymnasium as gym
from agents.dql import DQLAgent

import torch 


def make_env():


    pass

def main():
    seed = 42
    env = gym.make("BreakoutNoFrameskip-v4")
    agent = DQLAgent(env,is_deterministic=True)
    

    pass



if __name__ == '__main__':
    main()