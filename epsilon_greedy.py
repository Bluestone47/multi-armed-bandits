import numpy as np
from mab import MAB

class EpsGreedy(MAB):
    """
    Epsilon-Greedy multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    epsilon : float
        explore probability

    Q0 : float, optional
        initial value for the arms
    """

    def __init__(self, narms, epsilon, Q0=np.inf):
        self.narms = narms
        self.epsilon = epsilon
        self.action_attempts = np.zeros(narms)
        self.estimate_value = np.full(narms, Q0)

    def play(self, tround, context=None):
        if tround == 0 or np.random.random() < self.epsilon:
            arm = np.random.choice(self.narms)
        else:
            # estimate_value_max = np.where(np.isinf(self.estimate_value), -np.inf, self.estimate_value).argmax()
            arm_list = np.argwhere(self.estimate_value == np.amax(self.estimate_value))
            arm = np.random.choice(arm_list)
        return arm

    def update(self, arm, reward, context=None):
        self.action_attempts[arm] += 1
        q = (reward + self.estimate_value[arm] * (self.action_attempts[arm] - 1)) / self.action_attempts[arm]
        self.estimate_value[arm] = q



