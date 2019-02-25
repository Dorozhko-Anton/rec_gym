from agents.utils import Agent
import gin
import numpy as np

@gin.configurable
class RandomAgent(Agent):
    def __init__(self, action_size:int):
        self.action_size = action_size

    def begin_episode(self, observation):
        return self._select_action(observation)

    def step(self, reward, observation):
        return self._select_action(observation)

    def end_episode(self, reward):
        pass

    def bundle_and_checkpoint(self, directory, iteration):
        pass

    def unbundle(self, directory, iteration, dictionary):
        pass

    def _select_action(self, observation):
        state, items = observation
        n_items = len(items)
        return np.random.choice(list(range(n_items)), self.action_size)


@gin.configurable
class PopularityAgent(Agent):
    def begin_episode(self, observation):
        return super().begin_episode(observation)

    def step(self, reward, observation):
        return super().step(reward, observation)

    def end_episode(self, reward):
        return super().end_episode(reward)

    def bundle_and_checkpoint(self, directory, iteration):
        return super().bundle_and_checkpoint(directory, iteration)

    def unbundle(self, directory, iteration, dictionary):
        return super().unbundle(directory, iteration, dictionary)


# @gin.configurable
# class SVDpp(Agent):
#     pass


@gin.configurable
class LinUCB(Agent):
    def __init__(self,
                 d: int,
                 alpha: int = 0.2):

        self.d = d
        self.alpha = alpha

        # parameters of action by their id
        self.As = {}
        self.bs = {}

        self.last_action_id = None
        self.last_context = None

    def _choose_action(self, observation):
        user, items = observation

        x = user

        ps = np.zeros(len(items))
        pos2id = np.zeros(len(items))

        for idx, i in enumerate(items):
            if i.id not in self.As:
                self.As[i.id] = np.eye(self.d)
                self.bs[i.id] = np.zeros(self.d)

            A_inv = np.linalg.inv(self.As[i.id])
            theta = A_inv.dot(self.bs[i.id])
            ta = x.dot(A_inv).dot(x)
            a_upper_ci = self.alpha * np.sqrt(ta)
            a_mean = theta.dot(x)
            p = a_mean + a_upper_ci

            ps[idx] = p
            pos2id[idx] = i.id

        max_pos = np.argmax(ps)

        self.last_action_id = pos2id[max_pos]
        self.last_context = x

        return [pos2id[max_pos]]

    def _update_matrices(self, reward):
        if self.last_action_id is not None:
            self.As[self.last_action_id] += np.outer(self.last_context, self.last_context)
            self.bs[self.last_action_id] += reward * self.last_context

    def begin_episode(self, observation):
        return self._choose_action(observation)

    def step(self, reward, observation):
        # update matrix of previous action with reward
        self._update_matrices(reward)
        # select action from observation
        return self._choose_action(observation)

    def end_episode(self, reward):
        # update matrix of previous action with reward
        self._update_matrices(reward)

    def bundle_and_checkpoint(self, directory, iteration):
        # pickle all fields to directory
        bundle_dictionary = {'As': self.As,
                             'bs': self.bs,
                             'd': self.d,
                             'alpha': self.alpha}
        return bundle_dictionary

    def unbundle(self, directory, iteration, dictionary):
        # unpickle all fields from directory
        for key in self.__dict__:
            if key in dictionary:
                self.__dict__[key] = dictionary[key]

# ==============================

# =================================

from agents.baselines import Agent

class DotProdAgent(Agent):
    def __init__(self, n_rec):
        self.n_rec = n_rec

    def begin_episode(self, observation):
        user, items = observation
        scores = [user.dot(i) for i in items]
        idxs = np.argsort(scores)[::-1][:self.n_rec]
        return idxs

    def step(self, reward, observation):
        user, items = observation
        scores = [user.dot(i) for i in items]
        idxs = np.argsort(scores)[::-1][:self.n_rec]
        return idxs

    def end_episode(self, reward):
        raise NotImplemented


# =================================


from agents.baselines import Agent
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import dot, Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE


class MFAgent(Agent):
    def __init__(self, n_rec, state_dim, action_dim, sess):
        self.n_rec = n_rec
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sess = sess

        self.user_ph = Input(shape=(self.state_dim,), name='user')
        self.item_ph = Input(shape=(self.action_dim,), name='item')
        self.true_ph = Input(shape=(1,), name='true')

        net = Dense(20)(Concatenate()([self.user_ph, self.item_ph]))
        net = Dense(1)(net)
        self.rank_op = net

        self.loss = MSE(self.rank_op, self.true_ph)

        self.model = Model(inputs=[self.user_ph,
                                   self.item_ph],
                           outputs=self.rank_op)

        self.model.compile(loss=MSE, optimizer="adam")

        self.lr = 1e-4
        self.batch = 32
        self.n_iter = 10

        self.memory = []

        self.last_action = None
        self.last_user = None

    def _train(self):

        if len(self.memory) >= self.batch:
            for i in range(self.n_iter):
                batch_idxs = np.random.choice(range(len(self.memory)), size=self.batch)
                users, items, true_scores = zip(*[self.memory[i] for i in batch_idxs])

                self.model.fit([users, items, true_scores])

    def _get_actions(self, observation):
        user, items = observation

        scores = []
        for i in items:
            score = self.sess.run(self.rank_op, feed_dict={
                self.user_ph: user[None],
                self.item_ph: i[None],
            })
            scores.append(score[0][0])

        idxs = np.argsort(scores)[::-1][:self.n_rec]

        self.last_action = [items[i] for i in idxs]
        self.last_user = user
        return idxs

    def begin_episode(self, observation):
        self.sess.run(tf.global_variables_initializer())
        idxs = self._get_actions(observation)
        return idxs

    def step(self, reward, observation):
        for i, item in enumerate(self.last_action):
            self.memory.append([self.last_user, item, reward / self.n_rec])
        self.last_action = None
        self.last_user = None

        idxs = self._get_actions(observation)
        return idxs

    def end_episode(self, reward):
        raise NotImplemented


# =================================

#================================

# @gin.configurable
# class HLinUCB(Agent):
#     pass

from tensorflow.keras.layers import Input, Embedding, Flatten, Dot
from tensorflow.keras.models import Model

@gin.configurable
class DeepMF(Agent):
    def __init__(self, user_embedding_size, item_embedding_size, batch_size=32, train_steps=20):
        self.user_embedding_size = user_embedding_size
        self.item_embedding_size = item_embedding_size

        self.batch_size = batch_size
        self.train_steps = train_steps

        user_embedding = Input(shape=[self.user_embedding_size], name='user')
        item_embedding = Input(shape=[self.item_embedding_size], name='item')
        y = Dot(axes=1)([user_embedding, item_embedding])

        self.model = Model(inputs=[user_embedding, item_embedding], outputs=y)
        self.model.compile(optimizer='adam', loss='mae')

        # user, item, reward
        self.Xs = []
        self.ys = []

    def _select_action(self, observation):
        user, items = observation
        rewards = []

        for i in range(len(items)):
            # print(user)
            # print(user.shape)
            # print(items[i].embedding)
            # print(items[i].embedding.shape)
            r = self.model.predict([[user], [items[i].embedding]])
            rewards.append(r)

        idx = np.argmax(rewards)
        self.Xs.append(items[idx])
        return [idx]

    def _train(self):
        if len(self.memory) > self.batch_size:
            self.model.fit(x=self.Xs, y=self.ys,
                            batch_size=self.batch_size,
                               epochs=self.train_steps,
                               validation_split=0.1, shuffle=True)

    def _save_reward(self, reward):
        self.ys.append(reward)

    def begin_episode(self, observation):
        return self._select_action(observation)

    def step(self, reward, observation):
        self._save_reward(reward)

        return self._select_action(observation)

    def end_episode(self, reward):
        self._save_reward(reward)


    # def bundle_and_checkpoint(self, directory, iteration):
    #     return super().bundle_and_checkpoint(directory, iteration)
    #
    # def unbundle(self, directory, iteration, dictionary):
    #     return super().unbundle(directory, iteration, dictionary)
