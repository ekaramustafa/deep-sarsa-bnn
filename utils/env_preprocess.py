
from gymnasium import gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack,RecordVideo



class EnvironmentWrapper():

    def __init__(self,env) -> None:
        env = AtariPreprocessing(env,frame_skip=4,screen_size=84,grayscale_obs=True)
        env = FrameStack(env,num_stack=4)
        return env