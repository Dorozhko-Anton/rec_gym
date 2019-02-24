from collections import defaultdict, namedtuple

import gin
import gym
import numpy as np
from gym.utils import seeding
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA


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


Interaction = namedtuple('Interaction', ['t', 'uid', 'recs', 'rewards', 'probs', 'best_ps', 'ranks'])


@gin.configurable('PrimEnv2Ref')
class PrimEnv2Ref(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 n_items: int,
                 n_users: int,
                 n_rec: int,
                 embedding_dimension: int,
                 user_change_prob: float,
                 reward_noise: float,
                 user_init_sigma: float,
                 user_ar_coef: float,
                 user_drift_sigma: float,
                 initial_n_clusters: int,
                 cluster_var: float,
                 in_cluster_var: float,
                 new_items_interval: int,
                 new_items_size: float,
                 return_items_objects: bool,
                 click_prob_type: str = 'sigmoid'
                 ):

        self.seed()

        self.n_items = n_items
        self.n_users = n_users
        self.n_rec = n_rec
        self.embedding_dimension = embedding_dimension

        # user/items generation
        self.initial_n_clusters = initial_n_clusters
        self.cluster_var = cluster_var
        self.in_cluster_var = in_cluster_var
        self.user_init_sigma = user_init_sigma

        # reward
        self.reward_noise = reward_noise
        # click prob ['sigmoid', 'normal']
        self.click_prob_type = click_prob_type
        self.item_preference_var = 2

        # env evolution
        self.user_change_prob = user_change_prob
        self.user_ar_coef = user_ar_coef
        self.user_drift_sigma = user_drift_sigma
        self.new_items_interval = new_items_interval
        self.new_items_size = new_items_size
        self.user_type = 'drifting'  # TODO: make param

        # obs type
        self.return_items_objects = return_items_objects
        self.n_items_in_user_representation = 10

        # create items
        # create users
        self.user_counter = 0
        self.item_counter = 0
        self.users = self._load_users()
        self.items = self._load_items()
        self.active_user = self._load_active_user()

        self.steps_count = 0
        self.outdated_item_number = 0

        # recommendations at each step
        # relevancy as a probability of click for each object in recommendation at each step
        # reward for each product
        self.interactions = []
        self.bought_items = defaultdict(set)
        self.user_representations = defaultdict(list)

        self.recommendations = defaultdict(list)
        self.rewards = defaultdict(list)
        self.ps = defaultdict(list)

    def seed(self, seed=None):
        if seed is not None or not hasattr(self, 'main_seed'):
            self.main_seed = seed
            self.user_rng, self.user_seed = seeding.np_random(seed)
            self.userdrift_rng, self.userdrift_seed = seeding.np_random(seed)
            self.item_rng, self.item_seed = seeding.np_random(seed)
            self.reward_rng, self.reward_seed = seeding.np_random(seed)

        return [self.main_seed, self.user_seed, self.userdrift_seed, self.item_seed, self.reward_seed]

    def step(self, action):
        reward = self._compute_reward(action)
        current_user = self.active_user
        self._evolve()

        observations = self._get_observation()
        done = current_user != self.active_user
        info = None
        self.steps_count += 1
        return observations, reward, done, info

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
        users = {}
        for i in range(self.n_users):
            user_vec = self.user_init_sigma * self.user_rng.randn(self.embedding_dimension)
            u = User(id=self.user_counter, embedding=user_vec)
            self.user_counter += 1
            users[u.id] = u
        return users

    def _add_random_user(self):
        user_vec = self.user_init_sigma * self.user_rng.randn(self.embedding_dimension)
        u = User(id=self.user_counter, embedding=user_vec)
        self.user_counter += 1
        self.users[u.id] = u

    def _load_items(self):
        from sklearn.datasets import make_blobs

        cluster_centers = np.sqrt(self.cluster_var) * self.item_rng.randn(self.initial_n_clusters,
                                                                          self.embedding_dimension)

        item_vecs, clusters = make_blobs(n_samples=self.n_items,
                                         n_features=self.embedding_dimension,
                                         centers=cluster_centers,
                                         cluster_std=np.sqrt(self.in_cluster_var),
                                         # center_box=(-10.0, 10.0),
                                         shuffle=True,
                                         random_state=self.user_rng)

        items = {}
        for i in item_vecs:
            item = Item(id=self.item_counter, embedding=i, use_until=np.inf)
            self.item_counter += 1
            items[item.id] = item
        return items

    def _add_random_item(self):
        embedding = self.item_rng.randn(self.embedding_dimension) * self.cluster_var
        item = Item(id=self.item_counter, embedding=embedding, use_until=np.inf)
        self.item_counter += 1
        self.items[item.id] = item

    def _load_active_user(self):
        return np.random.choice(range(len(self.users)))

    # interactions
    def _order_proba(self, item):
        noise = self.reward_noise * (self.reward_rng.randn() - 0.5)
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

        return p + noise

    def _compute_reward(self, action_pos_ids):
        action = [self.item_pos2id[pos] for pos in action_pos_ids]

        ps = []
        rewards = []
        for a in action:
            # TODO: you can click only 1  on site
            p = self._order_proba(self.items[a].embedding)
            p = np.clip(p, 0, 1)
            # TODO : clip with baseline
            # p = np.clip(p, 0.1, 1)

            r = self.reward_rng.choice(
                [0, 1],
                p=[1 - p, p]
            )

            if r > 0:
                self.bought_items[self.active_user].add(self.items[a].id)
                self.user_representations[self.active_user].append(a)

            rewards.append(r)
            ps.append(p)

        # TODO: get all items p_click and save ranks of action items
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

        self.recommendations[self.active_user].append(action)
        self.rewards[self.active_user].append(rewards)
        self.ps[self.active_user].append(ps)
        self.interactions.append(Interaction(t=self.steps_count,
                                             uid=self.active_user,
                                             recs=action,
                                             rewards=rewards,
                                             probs=ps,
                                             best_ps=best_probas,
                                             ranks=ranks))

        return np.sum(rewards)

    def _change_user_preferences(self):
        if self.user_type == 'drifting':
            u = self.users[self.active_user].embedding

            # AR-1
            self.users[self.active_user].embedding = self.user_ar_coef * u + np.sqrt(
                1 - self.user_ar_coef ** 2) * self.userdrift_rng.randn(*u.shape) * self.user_drift_sigma

            # Random walk
            # self.users[self.active_user] = u + self.userdrift_rng.randn(*u.shape) * self.user_drift_sigma

            assert np.all(u == self.users[self.active_user].embedding), 'no drift?'

    def _choose_next_active_user(self):
        if self.n_users > 1:
            active_user_probas = np.ones(self.n_users) * self.user_change_prob / (self.n_users - 1)
            active_user_probas[self.active_user] = 1 - self.user_change_prob
            next_active_user = self.userdrift_rng.choice(range(self.n_users), p=active_user_probas)
            self.active_user = next_active_user

    def _evolve(self):
        # user preferences transition
        self._change_user_preferences()

        # choose next user for recommendations
        self._choose_next_active_user()

        # add new items
        if self.steps_count % self.new_items_interval == 0:
            n_items_to_add = int(self.new_items_size * self.n_items)
            for i in range(n_items_to_add):
                self._add_random_item()
            self.outdated_item_number += n_items_to_add

    def _get_possible_items(self):
        pos = 0
        self.item_pos2id = {}
        possible_items = []

        for k, item in self.items.items():
            if (self.steps_count < item.use_until) \
                    and (item.id > self.outdated_item_number) \
                    and item.id not in self.bought_items[self.active_user]:

                if self.return_items_objects:
                    possible_items.append(item)
                else:
                    possible_items.append(item.embedding)

                self.item_pos2id[pos] = item.id
                pos += 1

        self.possible_items = possible_items
        return possible_items

    def _get_observation(self):

        user_representation = self.users[self.active_user].embedding

        # last_bought_items = self.user_representations[self.active_user][-self.n_items_in_user_representation:]
        # user_representation = []
        # for i in range(self.n_items_in_user_representation-len(last_bought_items)):
        #     user_representation.extend(np.zeros(self.embedding_dimension))
        # for i in last_bought_items:
        #     user_representation.extend(self.items[i].embedding)


        # print(len(user_representation))
        return (np.array(user_representation), self._get_possible_items())
