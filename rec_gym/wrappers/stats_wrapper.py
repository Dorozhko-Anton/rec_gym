import gym
from collections import namedtuple

Interaction = namedtuple('Interaction', ['t', 'uid', 'recs', 'rewards', 'probs', 'best_ps', 'ranks'])


class StatsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.interactions = []
        self.t = 0
        self.last_uid = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        user, items = observation
        self.t += 1
        x = Interaction(t=self.t,
                        uid=self.last_uid,
                        rewards=reward,
                        probs=info.get('probs', None),
                        best_ps=info.get('best_ps', None),
                        ranks=info.get('ranks', None)
                        )
        self.interactions.append(x)
        self.last_uid = user.id
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        user, items = observation
        self.last_uid = user.id
        return observation
