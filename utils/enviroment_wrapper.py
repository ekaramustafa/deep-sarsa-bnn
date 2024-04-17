
from gymnasium.wrappers import AtariPreprocessing, FrameStack,RecordVideo

class EnvironmentWrapper():
    @staticmethod
    def wrap_environment(env):
        env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
        env = FrameStack(env, num_stack=4)
        return env