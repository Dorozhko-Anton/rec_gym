import numpy as np
from rec_gym.wrappers.dynamic_spaces_base_wrapper import DynamicSpacesWrapper
from collections import deque, defaultdict
import gym
from gym.spaces import Box, Tuple
import numpy as np


class ExplicitUserItemWrapper(DynamicSpacesWrapper):
    def __init__(self, env, maxlen=5):
        super().__init__(env)
        self.N = maxlen
        self.d = env.unwrapped.embedding_dimension
        self.K = env.unwrapped.n_rec

        def new_user():
            user = deque(maxlen=self.N)
            for i in range(self.N):
                user.append(np.zeros(self.d))
            return user

        self.user_repr = defaultdict(lambda: new_user())
        self.last_user = None
        self.last_items = None

        self.user_space = Box(low=-20, high=20, shape=(self.d,), dtype=np.float)
        self.hist_space = Tuple([Box(low=-20, high=20, shape=(self.d,), dtype=np.float) for i in range(self.N)])
        self.observation_space = Tuple(spaces=[self.user_space, self.hist_space])
        self.action_space = Box(low=-1, high=1, shape=(self.d,), dtype=np.float)

    def reset(self, **kwargs):
        observation = super().reset()
        self.last_user, self.last_items = observation

        self.observation_space = Tuple(spaces=[self.user_space, self.hist_space])
        self.action_space = Box(low=-1, high=1, shape=(self.d,), dtype=np.float)
        return self.creat_observation()

    def step(self, action):

        scores = [np.dot(action, i.embedding) for i in self.last_items]
        action = np.argsort(scores)[::-1][:self.K]

        observation, reward, done, info = super().step(action)

        if reward > 0:
            for a, r in zip(action, info['rewards']):
                if r > 0:
                    item = self.last_items[a].embedding
                    self.user_repr[self.last_user.id].append(item)

        done = self.last_user.id != observation[0].id

        self.last_user, self.last_items = observation

        self.observation_space = Tuple(spaces=[self.user_space, self.hist_space])
        self.action_space = Box(low=-1, high=1, shape=(self.d,), dtype=np.float)
        return self.creat_observation(), reward, done, info

    def creat_observation(self):
        user_hist = np.array([item for item in self.user_repr[self.last_user.id]])
        user = self.last_user.embedding
        return [user, user_hist]
