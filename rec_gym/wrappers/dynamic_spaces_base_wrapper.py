import gym


class DynamicSpacesWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _update_spaces(self):
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._update_spaces()
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self._update_spaces()
        return observation
