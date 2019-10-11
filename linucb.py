import numpy as np
from numpy.linalg import inv
from mab import MAB


class LinUCB(MAB):
    """
    Contextual multi-armed bandit (LinUCB)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    alpha : float
        positive real explore-exploit parameter
    """

    def __init__(self, narms, ndims, alpha):
        self.narms = narms
        self.ndims = ndims
        self.alpha = alpha
        self.inverse_covariance = []
        self.former_contexts = []
        self.upper_bounds = np.zeros(ndims)  # 1 * ndims
        self.total_rewards = np.zeros(narms)
        self.action_attempts = np.zeros(narms)
        self.estimate_value = np.full(narms, 0.0)
        for arm in range(self.narms):
            self.inverse_covariance.append(np.identity(self.ndims))  # ndims * ndims
            self.former_contexts.append(np.zeros((self.ndims, 1)))  # ndims * 1

    def play(self, tround, context):
        context = np.reshape(context, (self.narms, self.ndims))
        for arm in range(self.narms):
            context_arm = np.transpose(context[arm].reshape(1, self.ndims))
            theta_hat = inv(self.inverse_covariance[arm]) @ self.former_contexts[arm]  # ndims * 1
            np.seterr(divide='ignore')
            variance = np.transpose(context_arm) @ inv(self.inverse_covariance[arm]) @ context_arm
            std_dev = np.log(variance)  # ndims * ndims
            np.seterr(divide='warn')
            predicted_payoff = np.dot(np.transpose(theta_hat), context_arm)
            self.upper_bounds[arm] = predicted_payoff + self.alpha * std_dev
        arm_list = np.argwhere(self.upper_bounds == np.amax(self.upper_bounds)).flatten()
        arm = np.random.choice(arm_list)
        return arm

    def update(self, arm, reward, context):
        context = np.reshape(context, (self.narms, self.ndims))
        context_arm = np.transpose(context[arm].reshape(1, self.ndims))
        self.action_attempts[arm] += 1
        self.total_rewards[arm] += reward
        self.estimate_value[arm] = self.total_rewards[arm] / self.action_attempts[arm]
        self.inverse_covariance[arm] += np.dot(context_arm, np.transpose(context_arm))
        self.former_contexts[arm] = self.former_contexts[arm] + reward * context_arm
