from agents.utils import Agent
import numpy as np


class HLinUCB(Agent):
    def __init__(self,
                 d: int,
                 alpha: int = 0.2):

        self.d = d
        self.alpha = alpha

        self.k = 2 * d

        # parameters of action by their id
        self.As = {}
        self.Bs = {}
        self.bs = {}
        self.A0 = np.eye(self.k)
        self.b0 = np.zeros(self.k)

        self.last_action_id = None
        self.last_context = None

        self.eval_mode = False

    def _choose_action(self, observation):
        user, items = observation

        ps = np.zeros(len(items))
        pos2id = np.zeros(len(items))

        beta = np.linalg.inv(self.A0).dot(self.b0)

        for idx, i in enumerate(items):
            z = np.concatenate([user.embedding, i.embedding])
            x = i.embedding

            if i.id not in self.As:
                self.As[i.id] = np.eye(self.d)
                self.Bs[i.id] = np.zeros((self.d, self.k))
                self.bs[i.id] = np.zeros(self.d)

            A0_inv = np.linalg.inv(self.A0)
            A_inv = np.linalg.inv(self.As[i.id])
            Ba = self.Bs[i.id]
            ABtA = A0_inv.dot(Ba.T).dot(A_inv)

            s = z.dot(A0_inv).dot(z) - 2 * z.dot(ABtA).dot(x)
            s += x.dot(A_inv).dot(x) + x.dot(A_inv).dot(Ba).dot(ABtA).dot(x)

            theta = A_inv.dot(self.bs[i.id] - Ba.dot(beta))
            a_upper_ci = self.alpha * np.sqrt(s)
            a_mean = theta.dot(x) + beta.dot(z)
            p = a_mean + a_upper_ci

            ps[idx] = p
            pos2id[idx] = i.id

        max_pos = np.argmax(ps)

        self.last_action_id = int(pos2id[max_pos])
        self.last_context = items[max_pos].embedding
        self.last_z = np.concatenate([user.embedding, self.last_context])
        return [max_pos]

    def _update_matrices(self, reward):

        if self.last_action_id is not None:
            Ba = self.Bs[self.last_action_id]
            Aa_inv = np.linalg.inv(self.As[self.last_action_id])
            BAinv = Ba.T.dot(Aa_inv)

            self.A0 += BAinv.dot(Ba)
            self.b0 += BAinv.dot(self.bs[self.last_action_id])

            self.As[self.last_action_id] += np.outer(self.last_context, self.last_context)
            self.Bs[self.last_action_id] += np.outer(self.last_context, self.last_z)
            self.bs[self.last_action_id] += reward * self.last_context

            self.A0 += np.outer(self.last_z, self.last_z) - BAinv.dot(Ba)
            self.b0 += reward * self.last_z - BAinv.dot(self.bs[self.last_action_id])

    def begin_episode(self, observation):
        return self._choose_action(observation)

    def step(self, reward, observation, info=None):
        # update matrix of previous action with reward
        self._update_matrices(reward)
        # select action from observation
        return self._choose_action(observation)

    def end_episode(self, reward, info=None):
        # update matrix of previous action with reward
        self._update_matrices(reward)

    def bundle_and_checkpoint(self, directory, iteration):
        # pickle all fields to directory
        bundle_dictionary = {'As': self.As,
                             'Bs': self.Bs,
                             'bs': self.bs,
                             'd': self.d,
                             'alpha': self.alpha,
                             'A0': self.A0,
                             'b0': self.b0}
        return bundle_dictionary

    def unbundle(self, directory, iteration, dictionary):
        # unpickle all fields from directory
        for key in self.__dict__:
            if key in dictionary:
                self.__dict__[key] = dictionary[key]