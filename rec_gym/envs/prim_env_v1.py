import gym
import numpy as np
from scipy.special import expit as sigmoid
from collections import deque

from sklearn.datasets import make_blobs

env_1_args = {
    # TODO: change item set in time

    # number of clusters
    'K' : 4,
    # items per cluster # TODO: why we need that ?
    # 'n_k' : []

    # number of items n
    'num_items' : 100,
    # number of recommendations in each action L
    'num_recommendations' : 1,
    # number of users m
    'num_users' : 20,
    # embedding dimension - d
    'embedding_dimension' : 2,

    # cluster center var
    'cluster_var' : 16,
    # inner cluster var
    'in_cluster_var' : 0.5,

    # user leave probability  eps
    'active_user_change_proba' :  0.1,

    # reward Bernoulli noise var  \sigma^2
    'noise_sigma': 2,


    # TODO: make some kernels in env, and use name as parameter
    'user_drifting_kernel' : lambda x: x,
    'user_drift_autoreg_coef' : 0.1,
    'user_drift_sigma' : 4,
    # ['stubborn', 'drifting' , 'receptive]
    'user_type' : 'drifting', #[0.5, 0.3, 0.2],

    # seed of rng
    'seed' : 42,
}


class PrimEnv1(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        pass

    def init_gym(self, args):

        # set all key word arguments as attributes
        for key in args:
            setattr(self, key, args[key])

        self.rng = np.random.RandomState(seed=self.seed)

        self.users = self._load_users()
        self.items = self._load_items()
        self.active_user = self._load_active_user()

        self.time = 0
        self.user_drift_kernel = lambda x: x

        # for renderinf purposes
        self.last_actions = [deque(maxlen=5) for _ in range(self.num_users)]
        self.last_rewards = [deque(maxlen=5) for _ in range(self.num_users)]

    @property
    def _n_actions(self):
        return self.num_items

    def _load_users(self):
        users = self.rng.randn(self.num_users, self.embedding_dimension)
        return users

    def _load_items(self):
        #items = self.rng.randn(self.num_items, self.embedding_dimension)

        cluster_centers = np.sqrt(self.cluster_var)*self.rng.randn(self.K, self.embedding_dimension)

        items, clusters = make_blobs(n_samples=self.num_items,
                                       n_features=self.embedding_dimension,
                                       centers=cluster_centers,
                                       cluster_std=np.sqrt(self.in_cluster_var),
                                       #center_box=(-10.0, 10.0),
                                       shuffle=True,
                                       random_state=self.seed)
        print(items.shape)
        return items

    def _load_active_user(self):
        return np.random.choice(range(len(self.users)))


    def render(self, mode='human'):
        """
        use T-SNE to plot users, items and actions.
        :param mode:
        :return:
        """
        if mode == 'rgb_array':

            #from sklearn.manifold import TSNE # each call is different transformation
            from sklearn.decomposition import PCA
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            from matplotlib.figure import Figure
            from matplotlib import collections  as mc

            # build tsne dimention reduction if dims > 2
            if self.items.shape[1] > 2:
                if not hasattr(self, 'pca'):
                    self.pca = PCA(n_components=2)
                    self.pca.fit(self.items)

                items = self.pca.transform(self.items)
                users = self.pca.transform(self.users)
            else:
                items, users = self.items, self.users

            fig = Figure(figsize=(5, 5))
            canvas = FigureCanvas(fig)
            ax = fig.gca()

            ax.scatter(items[:, 0], items[:, 1], c='green', label='items')
            ax.scatter(users[:, 0], users[:, 1], c='red', label='users')

            # active user
            x, y  = users[self.active_user]
            ax.scatter(x, y, marker='*', c='black', s=20, label='active user')

            # active user recommendation history
            actions = self.last_actions[self.active_user]
            rewards = self.last_rewards[self.active_user]

            # TODO: if item set will change will have problems
            lines = [ [(x, y), (self.items[a][0], self.items[a][1])] for a in actions]
            c = [ 'yellow' if r else 'black' for r in rewards]
            lc = mc.LineCollection(lines, colors=c, linewidths=2)
            ax.add_collection(lc)

            ax.legend()
            ax.axis('off')
            canvas.draw()  # draw the canvas, cache the renderer

            width, height = [int(x) for x in fig.get_size_inches() * fig.get_dpi()]
            image = np.fromstring(canvas.tostring_rgb(),
                                dtype='uint8').reshape(height, width,3)

            return image
        else:
            pass


    def order_proba(self, item):
        eps = self.noise_sigma * self.rng.randn()
        return sigmoid(self.users[self.active_user].dot(item) + eps)

    def step(self, action):
        """

        :param action:
        :return:
        """
        if not isinstance(action, list):
            action = [action]

        assert self.active_user is not None, 'active user not set'

        # reward
        rewards = []
        for a in action:
            p = self.order_proba(self.items[a])
            r = self.rng.choice(
                [0, 1],
                p=[1-p, p]
            )
            rewards.append(r)

            # logging for rendering
            self.last_actions[self.active_user].append(a)
            self.last_rewards[self.active_user].append(r)

        reward = np.sum(rewards)



        # user preferences transition
        if self.user_type == 'drifting':
            u = self.users[self.active_user]

            # AR-1
            self.users[self.active_user] = (1 - self.user_drift_autoreg_coef) * u + self.user_drift_autoreg_coef * self.rng.randn(*u.shape)*self.user_drift_sigma

            # Random walk
            #self.users[self.active_user] = u + self.rng.randn(*u.shape) * self.user_drift_sigma

            assert np.all(u == self.users[self.active_user]), 'no drift?'

        if self.user_type == 'receptive':
            u = self.users[self.active_user]
            self.users[self.active_user] = self.user_preference_change(u, action, reward)

        # choose next user for recommendations
        active_user_probas = np.ones(self.num_users) * self.active_user_change_proba / (self.num_users - 1)
        active_user_probas[self.active_user] = 1 - self.active_user_change_proba
        next_active_user = self.rng.choice(range(self.num_users), p=active_user_probas)
        self.active_user = next_active_user

        observations = (self.users, self.items, self.active_user)

        done = False
        info = None
        self.time += 1
        return observations, reward, done, info

    def reset(self):
        pass