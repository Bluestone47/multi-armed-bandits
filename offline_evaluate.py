import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from epsilon_greedy import EpsGreedy
from ucb import UCB
from beta_thompson import BetaThompson


def offlineEvaluate(mab, arms, rewards, contexts, nrounds=None):
    """
    Offline evaluation of a multi-armed bandit

    Arguments
    =========
    mab : instance of MAB

    arms : 1D int array, shape (nevents,)
        integer arm id for each event

    rewards : 1D float array, shape (nevents,)
        reward received for each event

    contexts : 2D float array, shape (nevents, mab.narms*nfeatures)
        contexts presented to the arms (stacked horizontally)
        for each event.

    nrounds : int, optional
        number of matching events to evaluate `mab` on.

    Returns
    =======
    out : 1D float array
        rewards for the matching events
    """

    history = []  # history of arms
    payoff = []  # history of payoffs
    for t in range(nrounds):
        arm = mab.play(t)
        mab.update(arm, rewards[t], contexts[t])
        history.append(arm)
        out = mab.estimate_value
    return out


if __name__ == '__main__':

    arms = []
    rewards = []
    contexts = []
    dataset_file = open('dataset.txt', 'r')
    for line in dataset_file:
        event = line.split(' ')
        arms.append(event[0])
    rewards.append(event[1])
    contexts.append(event[2:-1])
    dataset_file.close()

    mab = EpsGreedy(10, 0.05)
    results_EpsGreedy = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('EpsGreedy average reward', np.mean(results_EpsGreedy))
