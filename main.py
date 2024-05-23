import random
import numpy as np
import gymnasium as gym
from utils.enviroment_wrapper import EnvironmentWrapper
from utils.parameters import ExplorationParameters, StepParameters
from layers.adin_vb import AdinVBL, AdinVBConv2d
from layers.alter_vb import AlterVBL, AlterVBConv2d
from agents.dql import DQLAgent
from agents.des import DESAgent
from agents.ds import DSAgent
from agents.ddes import DDESAgent
import torch 

import multiprocessing as mp

num_episodes = 500

def train_agent(env_name, agent_class, num_episodes,expl_params=None,step_param=None):
    env = gym.make(env_name)
    env = EnvironmentWrapper.wrap_environment(env)
    agent = agent_class(env, is_deterministic=False, linear_layer_class=AdinVBL, conv_layer_class=AdinVBConv2d,expl_params=expl_params,step_params=step_param)
    agent.train(num_episodes=num_episodes)
    agent.evaluate(num_episodes=num_episodes)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    perform_envs_des()
    # test_for_dqls()   


def test_for_dqls():
    print()
    print("=====================================")
    print("TEST FOR DQL STARTS")
    print()
    print("  BreakoutNoFrameskip-v4 DQL_AGENT") 
    train_agent("BreakoutNoFrameskip-v4",DQLAgent,num_episodes,None,None)
    print()
    print("  SpaceInvadersNoFrameskip-v4 DQL_AGENT")
    train_agent("SpaceInvadersNoFrameskip-v4",DQLAgent,num_episodes,None,None)
    print()
    print("  TennisNoFrameskip-v4 DQL_AGENT")
    train_agent("TennisNoFrameskip-v4",DQLAgent,num_episodes,None,None)
    print()
    print("=====================================")
    print()

def test_for_overestimation():
    
    print()
    print("=====================================")
    print("TEST FOR OVERESTIMATION STARTS")
    print("  SpaceInvadersNoFrameskip-v4 DDESAgent")
    train_agent("SpaceInvadersNoFrameskip-v4",DDESAgent,num_episodes,None,None)
    print()
    print("   BreakoutNoFrameskip-v4- DDESAgent")
    train_agent("BreakoutNoFrameskip-v4",DDESAgent,num_episodes,None,None)
    print("=====================================")
    print()

def perform_envs_des():
    processes = []
    environments = ["TennisNoFrameskip-v4"]
    for idx,environment in enumerate(environments):
        for agent_class in [DESAgent, DSAgent, DDESAgent]:
            train_agent(environment, agent_class, num_episodes,None,None)


def get_exp_parameters_cfg():
    param1 = ExplorationParameters(eps_start=0.9,eps_end=0.1,eps_decay=1500,name="long_high_expl")# High and Long exploration
    param2 = ExplorationParameters(eps_start=0.7,eps_end=0.1,eps_decay=2000,name="long_moderate_expl")#Moderate Exploration with Slower Decay
    param3 = ExplorationParameters(eps_start=0.9,eps_end=0.1,eps_decay=500, name="short_high_expl") # High and Short Exploration
    # param4 = Parameters(eps_start=0.7, eps_end=0.1, eps_decay=500, name="short_moderate_expl") # Moderate and Short Exploration
    # Note for report that param4 performed very poorly

    #    param2 = ExplorationParameters(eps_start=0.7,eps_end=0.1,eps_decay=2000,name="long_moderate_expl")#Moderate Exploration with Slower Decay
    #    is the best one so far
    return [param1,param2,param3]


def get_step_parameters_cfg():
    batch_sizes = [512,1024,2048,4096]
    gamma = 0.99
    tau = 0.005
    lr = 1e-4
    #param1 = StepParameters(batch_size=batch_sizes[0],gamma=gamma,tau=tau,lr=lr,name="batch_512")
    #param2 = StepParameters(batch_size=batch_sizes[1],gamma=gamma,tau=tau,lr=lr,name="batch_1024")
    param3 = StepParameters(batch_size=batch_sizes[2],gamma=gamma,tau=tau,lr=lr,name="batch_2048")
    param4 = StepParameters(batch_size=batch_sizes[3],gamma=gamma,tau=tau,lr=lr,name="batch_4096")
    return [param3,param4]

if __name__ == '__main__':
    main()