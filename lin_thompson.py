import numpy as np
from numpy.linalg import inv
from mab import MAB


class LinThompson(MAB):
    """
    Contextual Thompson sampled multi-armed bandit (LinThompson)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    v : float
        positive real explore-exploit parameter
    """

    def __init__(self, narms, ndims, v):
        self.narms = narms
        self.ndims = ndims
        self.v = v
        # self.inverse_covariance = []  # ndims * ndims
        # self.former_contexts = []  # ndims * 1
        # self.mu_hat = []
        self.inverse_covariance = np.identity(self.ndims)  # ndims * ndims
        self.mu_hat = np.zeros(self.ndims)  # ndims * 1
        self.former_contexts = np.zeros(self.ndims)  # ndims * 1
        self.total_rewards = np.zeros(narms)
        self.action_attempts = np.zeros(narms)
        self.estimate_value = np.full(narms, 0.0)

    def play(self, tround, context):
        # for arm in range(self.narms):
        #     content_arm = np.transpose(context[arm])
        #     if self.action_attempts[arm] == 0:
        #         self.inverse_covariance.append(np.identity(self.ndims))
        #         self.former_contexts.append(np.zeros((self.ndims, 1)))
        covariance = self.v**2 * inv(self.inverse_covariance)
        mu = np.random.multivariate_normal(self.mu_hat, covariance)
        context = np.reshape(context, (self.narms, self.ndims))
        posterior = np.dot(context, mu)  # narms * 1
        arm_list = np.argwhere(posterior == np.amax(posterior)).flatten()
        arm = np.random.choice(arm_list)
        return arm

    def update(self, arm, reward, context):
        context = np.reshape(context, (self.narms, self.ndims))
        content_arm = np.transpose(context[arm])
        self.action_attempts[arm] += 1
        self.total_rewards[arm] += reward
        self.estimate_value[arm] = self.total_rewards[arm] / self.action_attempts[arm]
        self.inverse_covariance += np.dot(content_arm, np.transpose(content_arm))
        self.former_contexts = self.former_contexts + reward * content_arm
        self.mu_hat = inv(self.inverse_covariance) @ self.former_contexts
