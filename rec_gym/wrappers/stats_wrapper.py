from collections import namedtuple
from rec_gym.wrappers.dynamic_spaces_base_wrapper import DynamicSpacesWrapper

Interaction = namedtuple('Interaction', ['t', 'uid', 'recs', 'rewards', 'probs', 'best_ps', 'ranks'])


class StatsWrapper(DynamicSpacesWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.interactions = []
        self.t = 0
        self.last_uid = None

    def step(self, action):
        observation, reward, done, info = super().step(action)
        user, items = observation
        self.t += 1
        x = Interaction(t=self.t,
                        uid=self.last_uid,
                        recs=info.get('recs', None),
                        rewards=info.get('rewards', None),
                        probs=info.get('probs', None),
                        best_ps=info.get('best_ps', None),
                        ranks=info.get('ranks', None)
                        )
        self.interactions.append(x)
        self.last_uid = user.id
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        user, items = observation
        self.last_uid = user.id
        return observation
