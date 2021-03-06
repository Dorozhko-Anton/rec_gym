from collections import defaultdict, namedtuple
import gin
import gym
import numpy as np
from sklearn.datasets import make_blobs

from sklearn.decomposition import PCA
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import scipy.stats

class User:
    def __init__(self,
                 id: int,
                 embedding: np.ndarray):
        self.id = id
        self.embedding = embedding


class Item:
    def __init__(self,
                 id: int,
                 embedding: np.ndarray,
                 use_until: int):
        self.id = id
        self.embedding = embedding
        self.use_until = use_until


Interaction = namedtuple('Interaction', ['t', 'uid', 'recs', 'rewards', 'probs'])


@gin.configurable('PrimEnv3')
class PrimEnv3(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 n_items: int,
                 n_users: int,
                 n_rec: int,
                 embedding_dimension: int,
                 cluster_var: float,
                 in_cluster_var: float,
                 user_change_prob: float,
                 reward_noise: float,
                 user_init_sigma: float,
                 user_ar_coef: float,
                 user_drift_sigma: float,
                 seed: int,
                 user_type: str,
                 new_items_interval: int,
                 new_items_size: float,
                 return_items_objects: bool,
                 user_graph_connectivity: float,
                 broadcast_std: float,
                 ):

        self.n_items = n_items
        self.n_users = n_users
        self.n_rec = n_rec
        self.embedding_dimension = embedding_dimension
        self.cluster_var = cluster_var
        self.in_cluster_var = in_cluster_var
        self.user_change_prob = user_change_prob
        self.reward_noise = reward_noise
        self.user_init_sigma = user_init_sigma
        self.user_ar_coef = user_ar_coef
        self.user_drift_sigma = user_drift_sigma
        self.random_seed = seed
        self.user_type = user_type
        self.rng = np.random.RandomState(seed=self.random_seed)
        self.rng_rewards = np.random.RandomState(seed=self.random_seed)
        self.return_items_objects = return_items_objects

        self.new_items_interval = new_items_interval
        self.new_items_size = new_items_size

        self.user_graph_connectivity = user_graph_connectivity

        # create items
        # create users
        self.user_counter = 0
        self.item_counter = 0
        self.users = self._load_users()
        self.items = self._load_items()
        self.active_user = self._load_active_user()

        self.time = 0
        self.outdated_item_number = 0

        # recommendations at each step
        # relevancy as a proba of click for each object in recommendation at each step
        # reward for each product
        self.recommendations = defaultdict(list)
        self.ps = defaultdict(list)
        self.rewards = defaultdict(list)
        self.interactions = []

        self.user_graph = defaultdict(set)
        self.bought_items = defaultdict(set)

        # keep 10 mean and stds around an item that user bought
        # if that user is a neighbour
        self.neighbourhood_influence = defaultdict(list)
        self.broadcast_std = broadcast_std

    def _load_users(self):
        users = {}
        for i in range(self.n_users):
            user_vec = self.user_init_sigma * self.rng.randn(self.embedding_dimension)
            u = User(id=self.user_counter, embedding=user_vec)
            self.user_counter += 1
            users[u.id] = u

        for i in range(self.n_users):
            neighbours = set(range(self.n_users))
            neighbours.remove(i)
            neighbours = list(neighbours)

            n_neighbours = int(self.user_graph_connectivity * len(neighbours))

            for j in np.random.choice(neighbours, size=n_neighbours):
                self.user_graph[i].append(j)


        return users

    def _load_items(self):
        cluster_centers = np.sqrt(self.cluster_var) * self.rng.randn(self.n_rec, self.embedding_dimension)

        item_vecs, clusters = make_blobs(n_samples=self.n_items,
                                         n_features=self.embedding_dimension,
                                         centers=cluster_centers,
                                         cluster_std=np.sqrt(self.in_cluster_var),
                                         # center_box=(-10.0, 10.0),
                                         shuffle=True,
                                         random_state=self.random_seed)

        items = {}
        for i in item_vecs:
            item = Item(id=self.item_counter, embedding=i, use_until=np.inf)
            self.item_counter += 1
            items[item.id] = item
        return items

    def _add_random_item(self):
        embedding = np.random.randn(self.embedding_dimension)*self.cluster_var
        item = Item(id=self.item_counter, embedding=embedding, use_until=np.inf)
        self.item_counter += 1
        self.items[item.id] = item

    def _add_random_user(self):
        user_vec = self.user_init_sigma * self.rng.randn(self.embedding_dimension)
        u = User(id=self.user_counter, embedding=user_vec)
        self.user_counter += 1
        self.users[u.id] = u

        neighbours = set(range(self.n_users)) - u.id
        n_neighbours = self.user_graph_connectivity * len(neighbours)
        for j in np.random.choice(neighbours, size=n_neighbours):
            self.user_graph[u.id].append(j)


    def _load_active_user(self):
        return np.random.choice(range(len(self.users)))

    def _order_proba(self, item):
        eps = self.reward_noise * self.rng.randn()
        # return sigmoid(self.users[self.active_user].dot(item) + eps)
        return np.exp(- (np.linalg.norm(self.users[self.active_user].embedding - item) + eps))

    def _compute_reward(self, action_pos_ids):
        action = [self.item_pos2id[pos] for pos in action_pos_ids]

        ps = []
        rewards = []
        for a in action:
            p = self._order_proba(self.items[a].embedding)

            # TODO: add influence proba
            influence_proba = np.mean([dist.pdf(self.item[a].embedding)
                                       for dist in self.neighbourhood_influence[self.active_user]])
            alpha = 0.7
            p = alpha*p + (1-alpha)*influence_proba
            p = np.clip(p, 0, 1)

            r = self.rng_rewards.choice(
                [0, 1],
                p=[1 - p, p]
            )

            if r > 0:
                self.bought_items[self.active_user].add(self.items[a].id)

                # TODO: broadcast the influence in the network
                for x in self.user_graph[self.active_user]:
                    mean = self.items[a].embedding
                    cov  = np.eye(len(mean)) * self.broadcast_std
                    dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
                    self.neighbourhood_influence[x].append(dist)


            rewards.append(r)
            ps.append(p)

        self.recommendations[self.active_user].append(action)
        self.rewards[self.active_user].append(rewards)
        self.ps[self.active_user].append(ps)
        self.interactions.append(Interaction(t=self.time, uid=self.active_user, recs=action, rewards=rewards, probs=ps))

        return np.sum(rewards)

    def _evolve(self):
        # user preferences transition
        if self.user_type == 'drifting':
            u = self.users[self.active_user].embedding

            # AR-1
            self.users[self.active_user].embedding = self.user_ar_coef * u + np.sqrt(
                1 - self.user_ar_coef ** 2) * self.rng.randn(*u.shape) * self.user_drift_sigma

            # Random walk
            # self.users[self.active_user] = u + self.rng.randn(*u.shape) * self.user_drift_sigma

            assert np.all(u == self.users[self.active_user].embedding), 'no drift?'

        # choose next user for recommendations
        active_user_probas = np.ones(self.n_users) * self.user_change_prob / (self.n_users - 1)
        active_user_probas[self.active_user] = 1 - self.user_change_prob
        next_active_user = self.rng.choice(range(self.n_users), p=active_user_probas)
        self.active_user = next_active_user

        # add new items
        if self.time % self.new_items_interval == 0:
            n_items_to_add = int(self.new_items_size * self.n_items)
            for i in range(n_items_to_add):
                self._add_random_item()

            self.outdated_item_number += n_items_to_add

    def _get_possible_items(self):
        pos = 0
        self.item_pos2id = {}
        possible_items = []

        for k, item in self.items.items():
            if (self.time < item.use_until) \
                    or (item.id > self.outdated_item_number) \
                    or item.id not in self.bought_items[self.active_user]:

                if self.return_items_objects:
                    possible_items.append(item)
                else:
                    possible_items.append(item.embedding)

                self.item_pos2id[pos] = item.id
                pos += 1
        return possible_items

    def step(self, action):
        reward = self._compute_reward(action)
        self._evolve()

        observations = (self.users[self.active_user].embedding, self._get_possible_items())
        done = False
        info = None
        self.time += 1
        return observations, reward, done, info

    def reset(self):
        """do nothing"""
        observations = (self.users[self.active_user].embedding, self._get_possible_items())
        return observations

    def render(self, mode='human'):
        """
        use PCA to plot users, items and actions.
        :param mode:
        :return:
        """
        if mode == 'rgb_array':
            users_vec = np.array([u.embedding for uid, u in self.users.items()])
            items_vec = np.array([i.embedding for iid, i in self.items.items()])

            # build dimensionality reduction if dims > 2
            if items_vec.shape[1] > 2:
                if not hasattr(self, 'pca'):
                    self.pca = PCA(n_components=2)
                    self.pca.fit(items_vec)

                items = self.pca.transform(items_vec)
                users = self.pca.transform(users_vec)
            else:
                items, users = items_vec, users_vec

            fig = Figure(figsize=(5, 5))
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            ax.scatter(items[:, 0], items[:, 1], c='green', label='items')
            ax.scatter(users[:, 0], users[:, 1], c='red', label='users')
            # active user
            x, y = users[self.active_user]
            ax.scatter(x, y, marker='*', c='black', s=20, label='active user')

            # active user recommendation history
            # actions = self.last_actions[self.active_user]
            # rewards = self.last_rewards[self.active_user]
            # TODO: if item set will change will have problems
            # if self.action_is_items:
            #     lines = [ [(x, y), a ] for a in actions]
            # else:
            #     lines = [ [(x, y), (self.items[a][0], self.items[a][1])] for a in actions]
            #
            # c = [ 'yellow' if r else 'black' for r in rewards]
            # lc = mc.LineCollection(lines, colors=c, linewidths=2)
            # ax.add_collection(lc)

            ax.legend()
            ax.axis('off')
            canvas.draw()  # draw the canvas, cache the renderer
            width, height = [int(x) for x in fig.get_size_inches() * fig.get_dpi()]
            image = np.fromstring(canvas.tostring_rgb(),
                                  dtype='uint8').reshape(height, width, 3)

            return image
        else:
            pass
