import numpy as np
from mab import MAB


class BetaThompson(MAB):
    """
    Beta-Bernoulli Thompson sampling multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    alpha0 : float, optional
        positive real prior hyperparameter

    beta0 : float, optional
        positive real prior hyperparameter
    """

    def __init__(self, narms, alpha0=1.0, beta0=1.0):
        self.narms = narms
        self.alpha = np.full(narms, alpha0)
        self.beta = np.full(narms, beta0)
        self.action_attempts = np.zeros(narms)
        self.estimate_value = np.zeros(narms)

    def play(self, tround, context=None):
        theta = []
        for i in range(self.narms):
            theta.append(np.random.beta(self.alpha[i], self.beta[i]))
        arm_list = np.argwhere(theta == np.amax(theta)).flatten()
        arm = np.random.choice(arm_list)
        return arm

    def update(self, arm, reward, context=None):
        self.action_attempts[arm] += 1
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        self.estimate_value[arm] = (self.alpha[arm] - 1) / self.action_attempts[arm]
