class Parameters:
    def __init__(self, name):
        self.name= name



class ExplorationParameters(Parameters):
    def __init__(self, eps_start, eps_end, eps_decay,name):
        super(ExplorationParameters, self).__init__(name)
        self.eps_start = eps_start 
        self.eps_end = eps_end
        self.eps_decay= eps_decay


class StepParameters(Parameters):
    def __init__(self, batch_size,gamma, tau, lr,name):
        super(StepParameters, self).__init__(name)
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.batch_size = self.batch_size