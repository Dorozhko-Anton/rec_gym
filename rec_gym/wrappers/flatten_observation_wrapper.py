import numpy as np
from rec_gym.wrappers.dynamic_spaces_base_wrapper import DynamicSpacesWrapper


class FlattenObservationsWrapper(DynamicSpacesWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observation = super().reset()
        return FlattenObservationsWrapper._flatten_observation(observation)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return FlattenObservationsWrapper._flatten_observation(observation), reward, done, info

    @staticmethod
    def _flatten_observation(observation):
        user, items = observation
        return user.embedding, np.array([item.embedding for item in items])