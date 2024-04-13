from agents.base import Agent

class DeepQLearning(Agent):

    def __init__(self, env, config):
        super(DeepQLearning, self).__init__(env, config)
        self.name = "Deep Q Learning Agent"
        self.init_message()

    