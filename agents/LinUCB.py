from agents.utils import Agent
import numpy as np

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

        self.eval_mode = False

    def _choose_action(self, observation):
        user, items = observation

        ps = np.zeros(len(items))
        pos2id = np.zeros(len(items))

        for idx, i in enumerate(items):

            x = i.embedding

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
        return [max_pos]

    def _update_matrices(self, reward):

        if self.last_action_id is not None:
            self.As[self.last_action_id] += np.outer(self.last_context, self.last_context)
            self.bs[self.last_action_id] += reward * self.last_context

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
                             'bs': self.bs,
                             'd': self.d,
                             'alpha': self.alpha}
        return bundle_dictionary

    def unbundle(self, directory, iteration, dictionary):
        # unpickle all fields from directory
        for key in self.__dict__:
            if key in dictionary:
                self.__dict__[key] = dictionary[key]