import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = []
        self._maxsize = size

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, action_tp1, done):
        """
        Make sure, _storage will not exceed _maxsize.
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        """
        data = (obs_t, action, reward, obs_tp1, action_tp1, done)

        # add data to storage
        self._storage.append(data)
        if len(self._storage) > self._maxsize:
            self._storage.pop(0)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        # <randomly generate batch_size integers to be used as indexes of samples>
        indexes = np.random.choice(range(len(self._storage)), size=batch_size)

        # collect <s,a,r,s',done> for each index
        states, actions, rewards, next_states, next_actions, is_done = zip(*[self._storage[i] for i in indexes])
        return np.array(states), np.array(actions), np.array(rewards), \
            np.array(next_states), np.array(next_actions), np.array(is_done)

    def save(self, checkpoint_dir, iteration_number):
        pass

    def load(self, checkpoint_dir, iteration_number):
        pass


# some metrics to consider
# http://bickson.blogspot.com/2012/10/the-10-recommender-system-metrics-you.html
# def precision_at_k(predict, target, k=5):
#     pass
#
#
# def ndcg_at_k(predict, target, k=5):
#     pass
#
#
# def diversity():
#     pass
#
#
# def novelty():
#     pass
#
#
# def recall():
#     pass
import matplotlib.pylab as plt
import ipywidgets as widgets
from collections import defaultdict
import seaborn as sns
import pandas as pd

def data_exploring_widget(env):
    data = env.unwrapped.interactions

    time_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(data),
        step=1,
        description='Time:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    def plot_counts(t):

        unwrapped_env = env.unwrapped
        n_users = len(unwrapped_env.users)
        n_items = len(unwrapped_env.items)

        counts_by_iid = defaultdict(int)
        user_item_counts = np.zeros((n_users, n_items))
        rewards_by_uid = defaultdict(list)
        all_rewards = []

        for interaction in data[:t]:
            for iid in interaction.recs:
                counts_by_iid[iid] += 1
                user_item_counts[interaction.uid][iid] += 1

            rewards_by_uid[interaction.uid].extend(interaction.rewards)
            all_rewards.extend(interaction.rewards)

        for uid in range(n_users):
            plt.plot(pd.Series(rewards_by_uid[uid]).rolling(window=20).mean(), label='user %s' % uid)

        plt.plot(pd.Series(all_rewards).rolling(window=20).mean(), label='Mean average precision @20')
        plt.legend()
        plt.title('moving ap@20 for each user')
        plt.show()

        plt.bar(counts_by_iid.keys(), counts_by_iid.values())
        plt.title('count of recommended items')
        plt.show()

        plt.figure(figsize=(30,6))
        sns.heatmap(data=user_item_counts)
        plt.xlabel('item')
        plt.ylabel('user')
        plt.title('user-item count heatmap')
        plt.show()


    interactive_plot = widgets.interactive(plot_counts, t=time_slider)
    return interactive_plot