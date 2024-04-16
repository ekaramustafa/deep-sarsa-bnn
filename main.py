import numpy as np
from networks.bnn import DQN_VBLinear
from layers.alter_vbl import AlterVBL
from layers.adin_vbl import AdinVBL
from test import FunctionApproxTester
from networks.dqn import DQNLinear, DQN_Conv2D

import torch 
def true_function(x):
    return torch.sin(x) + 0.5 * torch.cos(2 * x)

def main():
    print("Success")
    pass


def linear_test():
    input_dim = 1
    hidden_dim = 128
    output_dim = 1
    adin_model = DQN_VBLinear(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=output_dim,layer_class=AdinVBL)
    alter_model = DQN_VBLinear(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=output_dim,layer_class=AlterVBL)
    tester = FunctionApproxTester(42)
    true_func = true_function
    
    tester.test(adin_model,true_function=true_func,num_epochs=800)
    # tester.test(alter_model,true_function=true_func,num_epochs=800)

if __name__ == '__main__':
    main()