import random
import numpy as np
import gymnasium as gym
from utils.enviroment_wrapper import EnvironmentWrapper
from layers.adin_vb import AdinVBL, AdinVBConv2d
from layers.alter_vb import AlterVBL, AlterVBConv2d
from agents.dql import DQLAgent
from agents.des import DESAgent
from agents.ds import DSAgent
import torch 

import multiprocessing as mp


def train_agent(env_name, agent_class, num_episodes):
    env = gym.make(env_name)
    env = EnvironmentWrapper.wrap_environment(env)
    agent = agent_class(env, is_deterministic=True, linear_layer_class=AdinVBL, conv_layer_class=AdinVBConv2d)
    agent.train(num_episodes=num_episodes)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    seed = 42
    num_episodes = 1000
    processes = []
    agent_classes = [DESAgent, DSAgent]
    for agent_class in agent_classes:
        process = mp.Process(target=train_agent, args=("BreakoutNoFrameskip-v4", agent_class, num_episodes))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    pass


if __name__ == '__main__':
    main()