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
    out = []  # history of payoffs
    count = 0  # count of events
    for t in range(nrounds):
        while True:
            arm = mab.play(t)
            count += 1
            if count >= len(arms):
                return out  # reach the end of the logged dataset
            if count < len(arms) and arms[count] - 1 == arm:
                # print('###')
                # print(t)
                # print(count)
                # print(arm)
                break
        mab.update(arm, rewards[count], contexts[count])  # arm (0-9), arms (1-10)
        history.append(arm)
        out.append(rewards[count])
    # print(mab.alpha)
    # print(mab.beta)
    print(mab.action_attempts)
    print(mab.estimate_value)
    print(mab.upper_bounds)
    return out


if __name__ == '__main__':

    arms = []
    rewards = []
    contexts = []
    dataset_file = open('dataset.txt', 'r')
    for line in dataset_file:
        event = line.split(' ')[:-1]
        event = list(map(int, event))
        arms.append(event[0])
        rewards.append(event[1])
        contexts.append(event[2:])
    dataset_file.close()

    # mab = EpsGreedy(10, 0.05)
    # results_EpsGreedy = offlineEvaluate(mab, arms, rewards, contexts, 800)
    # print('EpsGreedy average reward', np.mean(results_EpsGreedy))

    mab = UCB(10, 1.0)
    results_UCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('UCB average reward', np.mean(results_UCB))

    # mab = BetaThompson(10, 1.0, 1.0)
    # results_BetaThompson = offlineEvaluate(mab, arms, rewards, contexts, 800)
    # print('BetaThompson average reward', np.mean(results_BetaThompson))
