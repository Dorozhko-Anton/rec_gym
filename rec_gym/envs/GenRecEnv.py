from collections import defaultdict, namedtuple
import gym
import numpy as np
from gym.utils import seeding

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from gym.spaces import Discrete
from rec_gym.spaces.ntuple_space import NDiscreteTuple
from .utils import User, Item


class GeneratedRecEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'user_preference.types' : ['static', 'drifting']
    }

    def __init__(self,
                 n_items: int,
                 n_users: int,
                 n_rec: int,
                 embedding_dimension: int,
                 user_change_prob: float,
                 reward_noise: float,
                 user_initial_n_clusters: int,
                 user_init_sigma: float,
                 user_cluster_sigma: float,
                 user_ar_coef: float,
                 user_drift_sigma: float,
                 initial_n_clusters: int,
                 cluster_var: float,
                 in_cluster_var: float,
                 new_items_interval: int,
                 new_items_size: float,
                 click_prob_type: str = 'sigmoid',
                 user_preference_type: str = 'static',
                 choose_only_one_item: bool = False,
                 ):

        self.seed()

        self.n_items = n_items
        self.n_users = n_users
        self.n_rec = n_rec
        self.embedding_dimension = embedding_dimension

        # user/items generation
        self.initial_n_clusters = initial_n_clusters
        self.cluster_sigma = cluster_var
        self.in_cluster_var = in_cluster_var

        self.user_init_sigma = user_init_sigma
        self.user_initial_n_clusters = user_initial_n_clusters
        self.user_cluster_sigma = user_cluster_sigma

        # reward
        self.reward_noise = reward_noise
        self.choose_only_one_item = choose_only_one_item
        # click prob ['sigmoid', 'normal']
        self.click_prob_type = click_prob_type
        self.item_preference_var = 2

        # env evolution
        self.user_change_prob = user_change_prob
        self.user_ar_coef = user_ar_coef
        self.user_drift_sigma = user_drift_sigma
        self.new_items_interval = new_items_interval
        self.new_items_size = new_items_size

        assert user_preference_type in self.metadata['user_preference.types'], \
            'Invalid user_preference_type. Should be one of %s' % self.metadata['user_preference.types']
        self.user_type = user_preference_type


        # create items
        # create users
        self.user_counter = 0
        self.item_counter = 0

        self.users = {}
        self._load_users()

        self.items = {}
        self._load_items()

        self.active_user = self._load_active_user()

        self.bought_items = defaultdict(set)
        # logs
        self.steps_count = 0
        self.info = {}

        # TODO: make action and observation space. checkout robotics envs + FlattenDictWrapper
        # https://github.com/openai/gym/tree/5404b39d06f72012f562ec41f60734bd4b5ceb4b/gym/envs/robotics
        self.action_space = None
        self.observation_space = None

    def seed(self, seed=None):
        if seed is not None or not hasattr(self, 'main_seed'):
            self.main_seed = seed
            self.user_rng, self.user_seed = seeding.np_random(seed)
            self.userdrift_rng, self.userdrift_seed = seeding.np_random(seed)
            self.item_rng, self.item_seed = seeding.np_random(seed)
            self.reward_rng, self.reward_seed = seeding.np_random(seed)

        return [self.main_seed, self.user_seed, self.userdrift_seed, self.item_seed, self.reward_seed]

    def step(self, action):
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
        return self._get_observation()

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

    # users and items
    def _load_users(self):
        """
        generate cluster centers and `self.n_users`
        :return:
        """
        self.user_cluster_centers = self.user_cluster_sigma * self.user_rng.randn(self.user_initial_n_clusters, self.embedding_dimension)
        for i in range(self.n_users):
            self._add_random_user()

    def _add_random_user(self):
        """
        generate random Users with mean as a random cluster center
        :return:
        """
        cluster_id = np.random.choice(self.user_initial_n_clusters)
        user_vec = self.user_init_sigma * self.user_rng.randn(self.embedding_dimension)
        user_vec += self.user_cluster_centers[cluster_id]
        u = User(id=self.user_counter, embedding=user_vec)
        self.user_counter += 1
        self.users[u.id] = u

    def _load_items(self):
        """
        generate Item clusters
        :return:
        """
        self.item_cluster_centers = self.cluster_sigma * self.item_rng.randn(self.initial_n_clusters, self.embedding_dimension)
        for i in range(self.n_items):
            self._add_random_item()

    def _add_random_item(self):
        embedding = self.item_rng.randn(self.embedding_dimension) * self.in_cluster_var
        cluster_id = self.item_rng.choice(self.initial_n_clusters)
        embedding += self.item_cluster_centers[cluster_id]

        item = Item(id=self.item_counter, embedding=embedding, use_until=np.inf)
        self.item_counter += 1
        self.items[item.id] = item

    def _load_active_user(self):
        """
        choice first user uniformly at random
        :return:
        """
        return np.random.choice(range(len(self.users)))

    # interactions
    def _order_proba(self, item):
        """
        compute click proba as an f(dot prod of user and item embeddings)

        f can be sigmoid or pdf of normal distribution

        add normal noise with parameters uniform [-self.reward_noise , +self.reward_noise]
        :param item:
        :return:
        """
        noise = self.reward_noise * (self.reward_rng.rand() - 0.5)

        if self.click_prob_type == 'sigmoid':
            from scipy.special import expit
            p = expit(self.users[self.active_user].embedding.dot(item))

        elif self.click_prob_type == 'normal':

            from scipy.stats import multivariate_normal
            if not hasattr(self, 'dist'):
                # self.dist = multivariate_normal(mean=np.zeros(len(item)),
                #                                cov=self.item_preference_var * np.eye(len(item)))
                self.dist = multivariate_normal(mean=0,
                                                cov=self.item_preference_var)

            x = np.linalg.norm(self.users[self.active_user].embedding - item)
            p = self.dist.pdf(x)
        else:
            raise Exception('no click_prob_type specified')

        return np.clip(p + noise, a_max=1, a_min=0)

    def _reward(self, action_pos_ids):
        action = [self.item_pos2id[pos] for pos in action_pos_ids]

        ps = []
        rewards = []
        # compute probs of click
        for a in action:
            p = self._order_proba(self.items[a].embedding)
            ps.append(p)

        for p in ps:
            r = self.reward_rng.binomial(n=1, p=p)
            if r > 0:
                self.bought_items[self.active_user].add(self.items[a].id)

            # if flag set
            # user can click only one item
            # and the she will interact with another recommendation
            if self.choose_only_one_item and np.sum(rewards) > 0:
                rewards.append(0)
            else:
                rewards.append(r)

        # get all items p_click and save ranks of action items
        all_ps = {self.items[i].id: self._order_proba(self.items[i].embedding)
                  for i in self.item_pos2id.values()}
        sorted_ps = sorted(all_ps.items(), key=lambda kv: kv[1])
        best_probas = [x[1] for x in sorted_ps[::-1][:len(action)]]
        ranks = []
        for a in action:
            for r, i in enumerate(sorted_ps[::-1]):
                if i[0] == a:
                    ranks.append(r + 1)
                    break

        self.info = {
            'rewards': rewards,
            'recs': action,
            'probs': ps,
            'best_ps': best_probas,
            'ranks': ranks,
        }
        return np.sum(rewards)

    def _change_user_preferences(self):
        """
        modify user embeddings
        :return:
        """
        if self.user_type == 'static':
            return

        if self.user_type == 'drifting':
            u = self.users[self.active_user].embedding

            # AR-1
            self.users[self.active_user].embedding = self.user_ar_coef * u + np.sqrt(
                1 - self.user_ar_coef ** 2) * self.userdrift_rng.randn(*u.shape) * self.user_drift_sigma

            # Random walk
            # self.users[self.active_user] = u + self.userdrift_rng.randn(*u.shape) * self.user_drift_sigma

            assert np.all(u == self.users[self.active_user].embedding), 'no drift?'

    def _choose_next_active_user(self):
        """
        change an active user with the probability `self.user_change_prob`
        :return:
        """
        if self.n_users > 1:
            active_user_probas = np.ones(self.n_users) * self.user_change_prob / (self.n_users - 1)
            active_user_probas[self.active_user] = 1 - self.user_change_prob
            next_active_user = self.userdrift_rng.choice(range(self.n_users), p=active_user_probas)
            self.active_user = next_active_user

    def _evolve(self):
        """
        model the dynamic of the environment
        :return:
        """
        # user preferences transition
        self._change_user_preferences()

        # choose next user for recommendations
        self._choose_next_active_user()

        # add new items
        # if self.steps_count % self.new_items_interval == 0:
        #     n_items_to_add = int(self.new_items_size * self.n_items)
        #     for i in range(n_items_to_add):
        #         self._add_random_item()
        #     self.outdated_item_number += n_items_to_add

    def _get_possible_items(self):
        pos = 0
        self.item_pos2id = {}
        possible_items = []

        for k, item in self.items.items():
            if (self.steps_count < item.use_until) \
                    and item.id not in self.bought_items[self.active_user]:

                possible_items.append(item)
                self.item_pos2id[pos] = item.id
                pos += 1

        self.possible_items = possible_items
        self.action_space = NDiscreteTuple(Discrete(len(possible_items)), self.n_rec)
        self.observation_space = None
        return possible_items

    def _get_observation(self):
        return self.users[self.active_user], self._get_possible_items()
