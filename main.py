import random
import numpy as np
import gymnasium as gym
from utils.enviroment_wrapper import EnvironmentWrapper
from utils.parameters import Parameters
from layers.adin_vb import AdinVBL, AdinVBConv2d
from layers.alter_vb import AlterVBL, AlterVBConv2d
from agents.dql import DQLAgent
from agents.des import DESAgent
from agents.ds import DSAgent
import torch 

import multiprocessing as mp

num_episodes = 500

def train_agent(env_name, agent_class, num_episodes,params=None):
    env = gym.make(env_name)
    env = EnvironmentWrapper.wrap_environment(env)
    agent = agent_class(env, is_deterministic=False, linear_layer_class=AlterVBL, conv_layer_class=AlterVBConv2d,params=params)
    agent.train(num_episodes=num_episodes)
    agent.evaluate(num_episodes=num_episodes)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    env = gym.make("TennisNoFrameskip-v4")
    # perform_param_tests()
    # perform_other_tests()


def perform_other_tests():
    processes = []
    environments = ["BreakoutNoFrameskip-v4","SpaceInvadersNoFrameskip-v4","TennisNoFrameskip-v4"]
    for idx,environment in enumerate(environments):
        process = mp.Process(target=train_agent, args=(environments[idx], DSAgent, num_episodes))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def perform_param_tests():
    processes = []
    agent_classes = [DESAgent] 
    params = get_exp_parameters_cfg()
    for idx,param in enumerate(params):
        process = mp.Process(target=train_agent, args=("BreakoutNoFrameskip-v4", agent_classes[0], num_episodes,param))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

def get_exp_parameters_cfg():
    param1 = Parameters(eps_start=0.9,eps_end=0.1,eps_decay=1500,name="long_high_expl")# High and Long exploration
    param2 = Parameters(eps_start=0.7,eps_end=0.1,eps_decay=2000,name="long_moderate_expl")#Moderate Exploration with Slower Decay
    param3 = Parameters(eps_start=0.9,eps_end=0.1,eps_decay=500, name="short_high_expl") # High and Short Exploration
    # param4 = Parameters(eps_start=0.7, eps_end=0.1, eps_decay=500, name="short_moderate_expl") # Moderate and Short Exploration
    # Note for report that param4 performed very poorly
    return [param1,param2,param3]

if __name__ == '__main__':
    main()