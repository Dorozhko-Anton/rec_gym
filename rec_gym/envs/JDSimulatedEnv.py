# TODO: simulated env from
# Zou, L., Xia, L., Ding, Z., Song, J., Liu, W., & Yin, D. (2019).
# Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems.
# Retrieved from http://arxiv.org/abs/1902.05570
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import permutations
from queue import PriorityQueue
from typing import Any

import gym
import numpy as np
from gym.spaces import Discrete
from gym.utils import seeding
from scipy.stats import entropy

from rec_gym.spaces.ntuple_space import NDiscreteTuple
from .utils import User, Item

from copy import deepcopy

@dataclass(order=True)
class PrioritizedUser:
    time: float
    user_id: Any = field(compare=False)


class JDSimulatedEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'diversity_engagement_type': ['static', 'drifting']
    }

    def __init__(self, n_users, n_items, embedding_size, recommendation_size, kappa, diversity_engagement_type,
                 a=1, b=0, V=1, d=1, mu=0, sigma=1, seed=None):
        """

        :param n_users: number of users
        :param n_items: number of items
        :param embedding_size: size of the user and item embeddings
        :param recommendation_size : number of items to recommend each step
        :param kappa: 1 - kappa is an importance of the main dimension
        :param diversity_engagement_type: ['linear', 'quadratic']
                 'linear' :    p(stay | session) = a*E + b,  a > 0
                               return_time       = V - d*E,  V > 0, d > 0
                 'quadratic' : p(stay | session) = exp( - (E - mu)**2 / sigma )
                               return_time       = V * (1 - exp( - (E - mu)**2 / sigma ) ), V > 0

        :param a:
        :param b:
        :param V:
        :param d:
        :param mu:
        :param sigma:
        :param seed random seed
        """
        # parameters
        self.M = n_users
        self.N = n_items
        self.dim = embedding_size
        self.recommendation_size = recommendation_size

        self.n_rec = recommendation_size
        self.embedding_dimension = self.dim

        assert self.recommendation_size > 0
        self.kappa = kappa
        assert 0 <= self.kappa < 1, 'kappa should be in [0, 1)'

        self.diversity_engagement_type = diversity_engagement_type
        assert self.diversity_engagement_type in ['linear', 'quadratic'], \
            'diversity_engagement_type should be one of %s' % JDSimulatedEnv.metadata['diversity_engagement_type']

        self.V = V
        assert self.V > 0, 'V should be positive'

        if self.diversity_engagement_type == 'linear':
            self.a = a
            assert self.a > 0, 'a should be positive'
            self.b = b
            self.d = d
            assert self.d > 0, 'd should be positive'

        if self.diversity_engagement_type == 'quadratic':
            self.mu = mu
            self.sigma = sigma
            assert self.sigma > 0, 'std should be positive'

        self.seed(seed)

        # data
        self.users = {}

        primary_user_topics = self.user_rng.choice(range(self.dim), self.M)
        for i, k in zip(range(self.M), primary_user_topics):
            v = self.user_rng.uniform(0, self.kappa, self.dim)
            v[k] = 1 - self.kappa
            v = v / np.linalg.norm(v)
            user = User(id=i, embedding=v)
            self.users[user.id] = user

        self.items = {}
        primary_item_topics = self.user_rng.choice(range(self.dim), self.N)
        for i, k in zip(range(self.N), primary_item_topics):
            v = self.user_rng.uniform(0, self.kappa, self.dim)
            v[k] = 1 - self.kappa
            v = v / np.linalg.norm(v)
            item = Item(id=i, embedding=v, use_until=np.inf)
            self.items[item.id] = item

        # queue of users prioritized by return time
        self.user_priority_queue = PriorityQueue(self.M)
        for user_id in self.users.keys():
            self.user_priority_queue.put(PrioritizedUser(time=0.,
                                                         user_id=user_id))
        next_user = self.user_priority_queue.get()
        self.current_time, self.active_user = next_user.time, next_user.user_id

        # user_id : set of item ids
        self.bought_items = defaultdict(set)

        # TODO: make action and observation space. checkout robotics envs + FlattenDictWrapper
        # https://github.com/openai/gym/tree/5404b39d06f72012f562ec41f60734bd4b5ceb4b/gym/envs/robotics
        self.action_space = None
        self.observation_space = None

        # metrics history
        self.stay_probability = []
        self.return_time = []

    def seed(self, seed=None):
        if seed is not None or not hasattr(self, 'main_seed'):
            self.main_seed = seed
            self.X_rng, self.X_seed = seeding.np_random(seed)
            self.click_rng, self.click_seed = seeding.np_random(seed)
            self.stay_rng, self.stay_seed = seeding.np_random(seed)
            self.user_rng, self.user_seed = seeding.np_random(seed)

        return [self.main_seed, self.X_seed, self.click_seed, self.stay_seed, self.user_rng]

    def step(self, action_pos_ids):
        action = [self.item_pos2id[pos] for pos in action_pos_ids]

        p_clicks = self._compute_p_clicks(action)
        p_stay, t_return = self._prob_to_stay_and_return_time(action)

        clicks = self.click_rng.rand(len(p_clicks)) < p_clicks
        for i, click in enumerate(clicks):
            if click:
                self.bought_items[self.active_user].add(action[i])

        if self.stay_rng.rand() < p_stay:
            # stay
            done = False

            possible_items = self._get_possible_items()
            if len(possible_items) < self.recommendation_size:
                next_user = self.user_priority_queue.get()
                self.current_time, self.active_user = next_user.time, next_user.user_id
        else:
            # done, change user, put current user in queue
            done = True
            possible_items = self._get_possible_items()
            if len(possible_items) > self.recommendation_size:
                self.user_priority_queue.put(PrioritizedUser(time=self.current_time,
                                                             user_id=self.active_user))
            next_user = self.user_priority_queue.get()
            self.current_time, self.active_user = next_user.time, next_user.user_id

        observation = self._get_observation()
        reward = np.sum(clicks)
        info = {
            'p_stay': p_stay,
            't_return': t_return,
            'p_clicks': p_clicks,
            'rewards' : clicks,
        }
        return observation, reward, done, info

    def reset(self):
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def _compute_p_clicks(self, action):
        return [self.users[self.active_user].embedding.dot(self.items[item_id].embedding)
                for item_id in action]

    def _prob_to_stay_and_return_time(self, action):
        E = self._mean_cross_kl(action)

        if self.diversity_engagement_type == 'linear':
            p_stay = self.a * E + self.b
            t_return = self.V - self.d * E
            return p_stay, t_return

        if self.diversity_engagement_type == 'quadratic':
            p_stay = np.exp(-(E - self.mu) ** 2 / self.sigma)
            t_return = self.V * (1 - p_stay)
            return p_stay, t_return

    def _mean_cross_kl(self, action):
        session = action
        # TODO: compute mean cross KL for a whole current session or only for current recommendation ?
        # session = self.current_session.extend(action)
        return np.mean([entropy(pk=self.items[i].embedding, qk=self.items[j].embedding)
                        for i, j in permutations(session, 2)])

    def _get_possible_items(self):
        pos = 0
        self.item_pos2id = {}
        possible_items = []

        for k, item in self.items.items():
            if item.id not in self.bought_items[self.active_user]:
                possible_items.append(item)
                self.item_pos2id[pos] = item.id
                pos += 1

        self.possible_items = possible_items
        self.action_space = NDiscreteTuple(Discrete(len(possible_items)), self.recommendation_size)
        self.observation_space = None
        return possible_items

    def _get_observation(self):
        user, items = self.users[self.active_user], self._get_possible_items()
        assert len(items) >= self.recommendation_size, 'bad item set'
        return user, items

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'user_priority_queue':
                q = PriorityQueue()
                q.queue = deepcopy(v.queue)
                setattr(result, k, q)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result
