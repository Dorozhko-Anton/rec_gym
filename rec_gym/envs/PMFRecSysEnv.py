import warnings
from collections import defaultdict

import gym
import numpy as np
from gym.spaces import Discrete
from gym.utils import seeding
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split

from rec_gym.spaces.ntuple_space import NDiscreteTuple
from rec_gym.envs.utils import User, Item


class MovieLens(gym.Env):
    def __init__(self,
                 embedding_dimension=20,
                 n_items_to_recommend=4,
                 seed=0,
                 n_users=40,
                 n_items=500,
                 normalize_reward=False):
        """
        Environment that models some sequential recommendation process by using MovieLens Dataset
        PMF (Probabilistic Matrix Factorization) is performed to obtain user/item embeddings

        :param embedding_dimension: size of the user/item embeddings
        :param n_items_to_recommend:  number of items to recommend actions is a list of that size
        :param seed:
        :param n_users: number of users
        :param n_items: number of items
        :param normalize_reward: normalize [1,5] ranks to [-1,1] rewards
        """
        self.normalize_reward = normalize_reward
        self.embedding_dimension = embedding_dimension
        self.n_rec = n_items_to_recommend
        self.seed(seed)
        # Load the movielens-100k dataset (download it if needed),
        data = Dataset.load_builtin('ml-100k')

        # sample random trainset and testset
        # test set is made of 25% of the ratings.
        self.trainset, self.testset = train_test_split(data, test_size=.25)

        self.algo = SVD(n_factors=self.embedding_dimension, biased=False)
        self.algo.fit(self.trainset)

        self.users = self.algo.pu[:n_users]
        self.items = self.algo.qi[:n_items]

        self.n_users = len(self.users)
        self.n_items = len(self.items)

        if self.n_users < n_users:
            warnings.warn("Only %d users are available in dataset" % self.n_users)
        if self.n_items < n_items:
            warnings.warn("Only %d items are available in dataset" % self.n_items)

        self.Users = {}
        for i in range(self.n_users):
            user = User(id=i, embedding=self.users[i])
            self.Users[user.id] = user

        self.Items = {}
        for j in range(self.n_items):
            item = Item(id=j, embedding=self.items[j], use_until=np.inf)
            self.Items[item.id] = item

        self.active_uid = self.np_random.choice(range(self.n_users))
        self.bought_items = defaultdict(set)
        # logs
        self.steps_count = 0
        self.info = {}

        # TODO: make action and observation space. checkout robotics envs + FlattenDictWrapper
        # https://github.com/openai/gym/tree/5404b39d06f72012f562ec41f60734bd4b5ceb4b/gym/envs/robotics
        self.action_space = None
        self.observation_space = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        """
        Get items available for recommendation for `self.active_uid` user
        i.e. the items that user didn't interacted with (didn't receive as a recommendation)

        :return: (user_repr, possible_items):
                user_repr - User
                possible_items - list of Items
        """
        pos = 0
        self.item_pos2id = {}
        possible_items = []

        for i in set(range(self.n_items)) - self.bought_items[self.active_uid]:
            possible_items.append(self.Items[i])
            self.item_pos2id[pos] = i
            pos += 1

        self.action_space = NDiscreteTuple(Discrete(len(possible_items)), self.n_rec)
        self.observation_space = None
        return self.Users[self.active_uid], possible_items

    def _reward(self, action):
        """
        Compute reward as scalar product of user and item embeddings obtained by PMF
        Normalize if `self.normalize_reward` is True

        :param action: array of indexes of size `self.n_rec` in possible items
        :return:
        """
        assert len(action) == self.n_rec
        uid = self.active_uid
        rewards = []
        iids = []
        for a in action:
            iid = self.item_pos2id[a]
            r = self.algo.estimate(u=uid, i=iid)
            if self.normalize_reward:
                r = 0.5 * (r - 3)
            rewards.append(r)
            self.bought_items[uid].add(iid)
            iids.append(iid)

        self.info = {
            'rewards': rewards,
            'recs': iids,
        }

        return np.sum(rewards)

    def _evolve(self):
        """
        Choose next active user at random uniformly between users who have possible items to recommend
        :return:
        """
        users_to_play = []
        for i in range(self.n_users):
            if len(self.bought_items[i]) < (self.n_items - self.n_rec + 1):
                users_to_play.append(i)

        if len(users_to_play) == 0:
            for i in range(self.n_users):
                print(len(self.bought_items[i]))
        self.active_uid = self.np_random.choice(users_to_play)

    def step(self, action):
        """

        :param action: array of indexes of size `self.n_rec` in possible items
        :return: observation: (user_repr, possible_items)
                 reward:  sum of scores for each item in the action
                 done:  always False
                 info:
        """
        self.steps_count += 1
        self.info = {}
        rewards = self._reward(action)

        reward = rewards
        self._evolve()
        observation = self._get_observation()
        done = None
        info = self.info
        return observation, reward, done, info

    def reset(self):
        """
        :return: initial observation
        """
        observation = self._get_observation()
        return observation

    def render(self, mode='human'):
        pass
