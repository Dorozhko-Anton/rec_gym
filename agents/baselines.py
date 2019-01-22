from agents.utils import Agent
import gin
import numpy as np

@gin.configurable
class RandomAgent(Agent):
    def __init__(self, action_size:int):
        self.action_size = action_size

    def begin_episode(self, observation):
        return self._select_action(observation)

    def step(self, reward, observation):
        return self._select_action(observation)

    def end_episode(self, reward):
        pass

    def bundle_and_checkpoint(self, directory, iteration):
        pass

    def unbundle(self, directory, iteration, dictionary):
        pass

    def _select_action(self, observation):
        state, items = observation
        n_items = len(items)
        return np.random.choice(list(range(n_items)), self.action_size)


@gin.configurable
class PopularityAgent(Agent):
    def begin_episode(self, observation):
        return super().begin_episode(observation)

    def step(self, reward, observation):
        return super().step(reward, observation)

    def end_episode(self, reward):
        return super().end_episode(reward)

    def bundle_and_checkpoint(self, directory, iteration):
        return super().bundle_and_checkpoint(directory, iteration)

    def unbundle(self, directory, iteration, dictionary):
        return super().unbundle(directory, iteration, dictionary)


@gin.configurable
class SVDpp(Agent):
    pass


@gin.configurable
class LinUCB(Agent):
    pass


@gin.configurable
class HLinUCB(Agent):
    pass


@gin.configurable
class PMF(Agent):
    pass