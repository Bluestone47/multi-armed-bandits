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
        self.max_upper_bound = 0
        self.action_attempts = np.zeros(narms)
        self.estimate_value = np.full(narms, Q0)

    def play(self, tround, context=None):
        self.tround = tround
        if tround == 0 or np.random.random() < 0.05:
            arm = np.random.choice(self.narms)
        else:
            arm_list = np.argwhere(self.estimate_value == np.amax(self.estimate_value))
            arm_list = [item for sublist in arm_list for item in sublist]  # make the nested list into a flat list
            arm = np.random.choice(arm_list)
        return arm

    def update(self, arm, reward, context=None):
        self.action_attempts[arm] += 1
        if np.isinf(self.estimate_value[arm]):
            u = reward
        else:
            u = (reward + self.estimate_value[arm] * (self.action_attempts[arm] - 1)) / self.action_attempts[arm]
        delta = self.rho * np.log(self.tround + 1) / self.action_attempts[arm]
        q = u + np.sqrt(delta)
        self.estimate_value[arm] = q
