import numpy as np
from mab import MAB


class UCB(MAB):
    """
    Upper Confidence Bound (UCB) multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    rho : float
        positive real explore-exploit parameter

    Q0 : float, optional
        initial value for the arms
    """

    def __init__(self, narms, rho, Q0=np.inf):
        self.narms = narms
        self.rho = rho
        self.tround = 0
        self.action_attempts = np.zeros(narms)
        self.estimate_value = np.full(narms, Q0)
        self.upper_bounds = np.full(narms, Q0)

    def play(self, tround, context=None):
        self.tround = tround
        arm_list = np.argwhere(self.upper_bounds == np.amax(self.upper_bounds)).flatten()
        arm = np.random.choice(arm_list)
        return arm

    def update(self, arm, reward, context=None):
        self.action_attempts[arm] += 1
        if np.isinf(self.estimate_value[arm]):
            u = reward
        else:
            u = (reward + self.estimate_value[arm] * (self.action_attempts[arm] - 1)) / self.action_attempts[arm]
        self.estimate_value[arm] = u
        boost = self.rho * np.log(self.tround) / self.action_attempts[arm]
        self.upper_bounds[arm] = self.estimate_value[arm] + np.sqrt(boost)
