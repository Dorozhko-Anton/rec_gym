from agents.utils import Agent
import numpy as np
from surprise import SVD, Reader, Dataset
import pandas as pd


class SVDAgent(Agent):
    def __init__(self, env):
        self.rec_size = env.n_rec
        self.env = env

        self.users = []
        self.items = []
        self.ratings = []

        self.__eval_mode = False

    def begin_episode(self, observation):
        if not self.eval_mode:
            return self._select_action_train(observation)
        return self._select_action_eval(observation)

    def step(self, reward, observation, info=None):
        if info:
            self._update_info(info)

        if not self.eval_mode:
            return self._select_action_train(observation)
        return self._select_action_eval(observation)

    def end_episode(self, reward, info=None):
        if info:
            self._update_info(info)

    #     def bundle_and_checkpoint(self, directory, iteration):
    #         pass

    #     def unbundle(self, directory, iteration, dictionary):
    #         pass

    @property
    def eval_mode(self):
        return self.__eval_mode

    @eval_mode.setter
    def eval_mode(self, x):
        self.train()
        self.__eval_mode = x

    def _update_info(self, info):
        uid = info['uid']
        for i, r in zip(info['recs'], info['rewards']):
            self.users.append(uid)
            self.items.append(i)
            self.ratings.append(r)

    def _select_action_train(self, observation):
        return self.env.get_ground_truth_action()

    def _select_action_eval(self, observation):
        user, items = observation
        item_ids = [i.id for i in items]
        uid = user.id

        scores = [self.algo.predict(uid, i).est for i in item_ids]

        return np.argsort(scores)[::-1][:self.rec_size]

    def train(self):

        print('Train SVD')
        # Creation of the dataframe. Column names are irrelevant.
        ratings_dict = {'itemID': self.items,
                        'userID': self.users,
                        'rating': self.ratings}
        df = pd.DataFrame(ratings_dict)

        # A reader is still needed but only the rating_scale param is requiered.
        reader = Reader(rating_scale=(1, 5))

        # The columns must correspond to user id, item id and ratings (in that order).
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        trainset = data.build_full_trainset()

        self.algo = SVD(n_factors=20)
        self.algo.fit(trainset)
