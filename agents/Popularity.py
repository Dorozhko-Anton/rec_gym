from agents.utils import Agent
import numpy as np
from collections import defaultdict


class PopularityAgent(Agent):
    modes = ['mean_rank', 'positive_clicks']

    def __init__(self, env, mode='mean_rank'):
        self.rec_size = env.n_rec
        self.env = env

        assert mode in PopularityAgent.modes, PopularityAgent.modes
        self.mode = mode

        self.items_clicks = defaultdict(lambda: 0)
        self.items_positive_clicks = defaultdict(lambda: 0)
        self.items_mean_ranks = defaultdict(lambda: -np.inf)

        self.eval_mode = False

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
    def _update_info(self, info):
        for i, r in zip(info['recs'], info['rewards']):
            if (r > 0):
                self.items_positive_clicks[i] += 1

            n = self.items_clicks[i]
            if n == 0:
                self.items_mean_ranks[i] = r
            else:
                self.items_mean_ranks[i] = (self.items_mean_ranks[i] * n + r) / (n + 1)

            self.items_clicks[i] += 1

    def _select_action_train(self, observation):
        return self.env.get_ground_truth_action()

    def _select_action_eval(self, observation):
        user, items = observation
        item_ids = [i.id for i in items]

        clicks = [self.items_positive_clicks[i] for i in item_ids]
        ranks = [self.items_mean_ranks[i] for i in item_ids]

        if self.mode == 'mean_rank':
            return np.argsort(ranks)[::-1][:self.rec_size]

        return np.argsort(clicks)[::-1][:self.rec_size]

