import warnings
from collections import defaultdict
import pandas as pd
import gym
import numpy as np
from gym.spaces import Discrete
from gym.utils import seeding
from surprise import Dataset, SVD, Reader

from rec_gym.spaces.ntuple_space import NDiscreteTuple
from rec_gym.envs.utils import User, Item
import cloudpickle
import os


class MovieLensDRR(gym.Env):
    def __init__(self,
                 embedding_dimension=20,
                 n_items_to_recommend=1,
                 env_seed=0,
                 normalize_reward=False,
                 filename="/home/anton/Datasets/MovieLens/ml-100k/u.data",
                 sep='\t',
                 session_time=None,
                 session_size=None,
                 cache_dir='~/cache/',
                 shuffle_sessions=False,
                 ):
        """
        Environment that models some sequential recommendation process by using MovieLens Dataset
        PMF (Probabilistic Matrix Factorization) is performed to obtain user/item embeddings

        :param embedding_dimension: size of the user/item embeddings
        :param n_items_to_recommend:  number of items to recommend actions is a list of that size
        :param seed:
        :param normalize_reward: normalize [1,5] ranks to [-1,1] rewards
        :param dataset_name: 'ml-100k' , 'ml-1m'
        """
        self.normalize_reward = normalize_reward
        self.embedding_dimension = embedding_dimension
        self.n_rec = n_items_to_recommend
        self.seed(env_seed)
        self.shuffle_sessions = shuffle_sessions

        assert (session_size is None) != (session_time is None), 'session_size and session_time cannot be both not None'

        if session_time is not None:
            cache_name = '_'.join([str(embedding_dimension), 't', str(session_time), str(env_seed)] + filename.split('/')[-2:])
        elif session_size is not None:
            cache_name = '_'.join([str(embedding_dimension), 's', str(session_size), str(env_seed)] + filename.split('/')[-2:])

        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        cache_dir_name = os.path.join(cache_dir, cache_name)

        if not os.path.exists(cache_dir_name):
            os.mkdir(cache_dir_name)

            self.filename = filename
            data_columns = ['User', 'Item', 'Rating', 'Timestamp']
            data = pd.read_csv(self.filename, sep=sep, names=data_columns)
            reader = Reader(rating_scale=(1, 5))
            dataset = Dataset.load_from_df(data[['User', 'Item', 'Rating']], reader)
            #
            self.trainset = dataset.build_full_trainset()

            self.algo = SVD(n_factors=self.embedding_dimension, biased=False)
            self.algo.fit(self.trainset)
            # BEGIN session split
            if session_time:
                sessions = data.groupby(by='User').apply(
                    lambda x: (x.sort_values('Timestamp')['Timestamp'].diff() > session_time).cumsum(skipna=False))
                data['Session'] = sessions.droplevel(0)
            elif session_size:

                if shuffle_sessions:
                    sessions = data.sample(frac=1., random_state=env_seed).groupby(by='User').apply(
                        lambda x: pd.Series(data=np.arange(len(x)) // session_size, index=x.index))
                    data['Session'] = sessions.droplevel(0)
                else:
                    sessions = data.sort_values('Timestamp').groupby(by='User').apply(
                        lambda x: pd.Series(data=np.arange(len(x)) // session_size, index=x.index))
                    data['Session'] = sessions.droplevel(0)

            self.session_items = defaultdict(list)
            self.session_rewards = defaultdict(list)
            self.users_order = []

            for (uid, sid), session in data.groupby(by=['User', 'Session']):
                # convert raw uids to inner uids
                uid = self.algo.trainset.to_inner_uid(uid)
                # convert raw iids to inner iids
                s_items = session['Item'].values
                s_items = [self.algo.trainset.to_inner_iid(x) for x in s_items]

                s_ratings = session['Rating'].values

                self.users_order.append(uid)
                self.session_items[uid].append(s_items)
                self.session_rewards[uid].append(s_ratings)
                # END session split
            with open(os.path.join(cache_dir_name, 'data'), 'wb') as f:
                cloudpickle.dump([self.algo,
                                  self.users_order,
                                  self.session_items,
                                  self.session_rewards,
                                  ], f)

        else:
            print('User cache %s' % cache_dir_name)
            # Cache
            # self.algo
            # self.users_order
            # self.session_items
            # self.session_rewards
            with open(os.path.join(cache_dir_name, 'data'), 'rb') as f:
                self.algo, self.users_order, self.session_items, self.session_rewards = cloudpickle.load(f)

        if shuffle_sessions:
            print('shuffle sessions')
            np.random.seed(env_seed)
            np.random.shuffle(self.users_order)
            for uid, sessions in self.session_items.items():
                np.random.shuffle(sessions)
                self.session_items[uid] = sessions

        self.train_ranks = defaultdict(dict)
        for u, i, r in self.algo.trainset.all_ratings():
            self.train_ranks[u][i] = r

        self.users_session_number = defaultdict(int)
        self.users = self.algo.pu
        self.items = self.algo.qi

        self.n_users = len(self.users)
        self.n_items = len(self.items)

        self.Users = {}
        for i in range(self.n_users):
            user = User(id=i, embedding=self.users[i])
            self.Users[user.id] = user

        self.Items = {}
        for j in range(self.n_items):
            item = Item(id=j, embedding=self.items[j], use_until=np.inf)
            self.Items[item.id] = item

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

        for i in self.curr_possible_list:
            possible_items.append(self.Items[i])
            self.item_pos2id[pos] = i
            pos += 1

        self.action_space = NDiscreteTuple(Discrete(len(possible_items)), self.n_rec)
        self.observation_space = None
        return self.Users[self.curr_user], possible_items

    def _reward(self, action):
        """
        Compute reward as scalar product of user and item embeddings obtained by PMF
        Normalize if `self.normalize_reward` is True

        :param action: array of indexes of size `self.n_rec` in possible items
        :return:
        """
        assert len(action) == self.n_rec
        uid = self.curr_user
        rewards = []
        iids = []

        def normalize(r):
            return 0.5 * (r - 3)

        for a in action:
            iid = self.item_pos2id[a]

            if uid in self.train_ranks and iid in self.train_ranks[uid]:
                r = self.train_ranks[uid][iid]
            else:
                r = self.algo.estimate(u=uid, i=iid)

            if self.normalize_reward:
                r = normalize(r)

            rewards.append(r)
            iids.append(iid)

            self.curr_possible_list.remove(iid)

        gt_items = self.session_items[self.curr_user][self.curr_user_session_number][
                   self.in_session_pos:self.in_session_pos + self.n_rec]
        gt_rewards = self.curr_reward_list[self.in_session_pos:self.in_session_pos + self.n_rec]
        norm_gt_rewards = [normalize(r) for r in gt_rewards]
        self.info = {
            'rewards': rewards,
            'recs': iids,
            'ground_truth_items': gt_items,
            'ground_truth_rewards': gt_rewards,
            'norm_truth_rewards': norm_gt_rewards,
            'uid': uid,
        }
        self.in_session_pos += self.n_rec
        return np.sum(rewards)

    def get_ground_truth_action(self, emb=False):
        gt_items = self.session_items[self.curr_user][self.curr_user_session_number][
                   self.in_session_pos:self.in_session_pos + self.n_rec]
        id2pos = {v: k for k, v in self.item_pos2id.items()}
        #         print(id2pos)
        if emb:
            return self.Items[gt_items[0]]

        return [id2pos[x] for x in gt_items]



    def step(self, action):
        """

        :param action: array of indexes of size `self.ncache_rec` in possible items
        :return: observation: (user_repr, possible_items)
                 reward:  sum of scores for each item in the action
                 done:  always False
                 info:
        """
        self.steps_count += 1
        self.info = {}
        reward = self._reward(action)

        done = len(self.curr_possible_list) < self.n_rec
        if done:
            observation = None
        else:
            observation = self._get_observation()
        info = self.info
        return observation, reward, done, info

    def reset(self):
        """
        :return: initial observation
        """
        if (self.steps_count == 0):
            # CURRENT STATE
            self.curr_user_pos = 0
            self.curr_user = self.users_order[self.curr_user_pos]
            self.curr_user_session_number = self.users_session_number[self.curr_user]
            self.curr_possible_list = set(self.session_items[self.curr_user][self.curr_user_session_number])
            self.curr_reward_list = self.session_rewards[self.curr_user][self.curr_user_session_number]

            self.in_session_pos = 0
        else:
            while True:
                self.users_session_number[self.curr_user] += 1

                self.curr_user_pos += 1
                self.curr_user = self.users_order[self.curr_user_pos]
                self.curr_user_session_number = self.users_session_number[self.curr_user]
                self.curr_possible_list = set(self.session_items[self.curr_user][self.curr_user_session_number])
                self.curr_reward_list = self.session_rewards[self.curr_user][self.curr_user_session_number]
                self.in_session_pos = 0
                if (len(self.curr_possible_list) >= self.n_rec): break

        observation = self._get_observation()
        return observation

    def render(self, mode='human'):
        pass
