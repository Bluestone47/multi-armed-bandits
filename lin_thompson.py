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
        self.inverse_covariance = np.empty(narms, dtype=object)  # ndims * ndims
        self.former_contexts = np.empty(narms, dtype=object)  # ndims * 1
        self.total_rewards = np.zeros(narms)
        self.action_attempts = np.zeros(narms)
        self.estimate_value = np.full(narms, 0.0)

    def play(self, tround, context):
        return 0

    def update(self, arm, reward, context):
        print()
