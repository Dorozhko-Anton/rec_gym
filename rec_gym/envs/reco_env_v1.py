import gym
import numpy as np
from scipy.special import expit as sigmoid

env1_args = {
    # TODO: change item set in time
    'num_items' : 100,  # n
    'num_recommendations' : 1, # L
    'num_users' : 20, # m
    'embedding_dimention' : 10, # d

    'active_user_change_proba' :  0.1, # eps

    # TODO: make some kernels in env, and use name as parameter
    'user_drifting_kernel' : lambda x: x,
    'user_drift_autoreg_coef' : 0.1,

    # ['stubborn', 'drifting' , 'receptive']
    'user_type' : [0.5, 0.3, 0.2],

    'noise_sigma' : 2, # \sigma^2

    'seed' : 42, # seed of rng
}

class PrimEnv1(gym.Env):
    def __init__(self):
        """

        """
        self.users = None
        self.items = None
        self.time = 0
        self.active_user = None
        self.user_drift_kernel = lambda x: x

        self.rng = np.random.RandomState(seed=env1_args['seed'])
        pass

    def render(self, mode='human'):
        """
        use T-SNE to draw users, items and actions.

        :param mode:
        :return:
        """
        pass

    def order_proba(self, action):
        eps = env1_args['noise_sigma'] * self.rng.randn()
        return sigmoid(self.users[self.active_user].dot(action) + eps)

    def step(self, action):
        """

        :param action:
        :return:
        """

        # reward
        rewards = []
        for a in action:
            p = self.order_proba(a)
            r = self.rng.choice(
                [0, 1],
                p=[1-p, p]
            )
            rewards.append(r)
        reward = np.sum(rewards)

        # user preferences transition
        if env1_args['user_type'] == 'drifting':
            u = self.users[self.active_user]
            self.users[self.active_user] = (1-env1_args['user_drift_autoreg_coef'])*u + env1_args['user_drift_autoreg_coef']*self.rng.randn(*u.shape)

        if env1_args['user_type'] == 'receptive':
            u = self.users[self.active_user]
            self.users[self.active_user] = self.user_preference_change(u, action, reward)

        # choose next user for recommendations
        active_user_probas = np.ones_like(self.users.keys()) * env1_args['active_user_change_proba'] / (env1_args['num_users'] - 1)
        active_user_probas[self.active_user] = 1 - env1_args['active_user_change_proba']
        next_active_user = self.rng.choice(self.users.keys(), p=active_user_probas)
        self.active_user = next_active_user

        observations = (self.users, self.items, self.active_user)

        done = False
        info = None

        return observations, reward, done, info

    def reset(self):
        pass