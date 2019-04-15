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


class SumTree(object):
        """
        This SumTree code is modified version of Morvan Zhou:
        https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
        """
        """
        Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
        """
        def __init__(self, capacity):
            self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences
            self.data_pointer = 0
            self.full = 0

            # Generate the tree with all nodes values = 0
            # To understand this calculation (2 * capacity - 1) look at the schema above
            # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
            # Parent nodes = capacity - 1
            # Leaf nodes = capacity
            self.tree = np.zeros(2 * capacity - 1)

            """ tree:
                0
               / \
              0   0
             / \ / \
            0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
            """

            # Contains the experiences (so the size of data is capacity)
            self.data = np.zeros(capacity, dtype=object)

        def __len__(self):
            return self.capacity if self.full else self.data_pointer

        """
        Here we add our priority score in the sumtree leaf and add the experience in data
        """

        def add(self, priority, data):
            # Look at what index we want to put the experience
            tree_index = self.data_pointer + self.capacity - 1

            """ tree:
                0
               / \
              0   0
             / \ / \
    tree_index  0 0  0  We fill the leaves from left to right
            """

            # Update data frame
            self.data[self.data_pointer] = data

            # Update the leaf
            self.update(tree_index, priority)

            # Add 1 to data_pointer
            self.data_pointer += 1

            if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
                self.data_pointer = 0
                if not self.full:
                    self.full = 1

        """
        Update the leaf priority score and propagate the change through tree
        """

        def update(self, tree_index, priority):
            # Change = new priority score - former priority score
            change = priority - self.tree[tree_index]
            self.tree[tree_index] = priority

            # then propagate the change through tree
            while tree_index != 0:  # this method is faster than the recursive loop in the reference code

                """
                Here we want to access the line above
                THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                    0
                   / \
                  1   2
                 / \ / \
                3  4 5  [6] 

                If we are in leaf at index 6, we updated the priority score
                We need then to update index 2 node
                So tree_index = (tree_index - 1) // 2
                tree_index = (6-1)//2
                tree_index = 2 (because // round the result)
                """
                tree_index = (tree_index - 1) // 2
                self.tree[tree_index] += change

        """
        Here we get the leaf_index, priority value of that leaf and experience associated with that index
        """

        def get_leaf(self, v):
            """
            Tree structure and array storage:
            Tree index:
                 0         -> storing priority sum
                / \
              1     2
             / \   / \
            3   4 5   6    -> storing priority for experiences
            Array type for storing:
            [0,1,2,3,4,5,6]
            """
            parent_index = 0

            while True:  # the while loop is faster than the method in the reference code
                left_child_index = 2 * parent_index + 1
                right_child_index = left_child_index + 1

                # If we reach bottom, end the search
                if left_child_index >= len(self.tree):
                    leaf_index = parent_index
                    break

                else:  # downward search, always search for a higher priority node

                    if v <= self.tree[left_child_index]:
                        parent_index = left_child_index

                    else:
                        v -= self.tree[left_child_index]
                        parent_index = right_child_index

            data_index = leaf_index - self.capacity + 1

            return leaf_index, self.tree[leaf_index], self.data[data_index]

        @property
        def total_priority(self):
            return self.tree[0]  # Returns the root node


class PER(object):
    # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """

    def __init__(self, capacity, e=0.01, a= 0.6, b = 0.4, b_increment = 0.001, abs_error_upper = 1.):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)

        self.PER_e = e # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = a  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = b  # importance-sampling, from initial value increasing to 1
        self.PER_b_increment_per_sampling = b_increment
        self.absolute_error_upper = abs_error_upper  # clipped abs error

    def __len__(self):
        return len(self.tree)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = data

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class Agent:
    def begin_episode(self, observation):
        raise NotImplemented

    def step(self, reward, observation):
        raise NotImplemented

    def end_episode(self, reward):
        raise NotImplemented

    def bundle_and_checkpoint(self, directory, iteration):
        raise NotImplemented

    def unbundle(self, directory, iteration, dictionary):
        raise NotImplemented





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

        count_of_recs_for_uid = defaultdict(int)

        for interaction in data[:t]:
            for iid in interaction.recs:
                counts_by_iid[iid] += 1
                user_item_counts[interaction.uid][iid] += 1

            count_of_recs_for_uid[interaction.uid] += 1

            rewards_by_uid[interaction.uid].extend(interaction.rewards)
            all_rewards.extend(interaction.rewards)

        print('Mean reward:', np.mean(all_rewards))

        c = defaultdict(int)
        for i in env.interactions:
            c[i.uid] += 1

        cb = defaultdict(int)
        for k, v in env.bought_items.items():
            cb[k] = len(v)


        plt.bar(c.keys(), c.values(), label='recommended')
        plt.bar(cb.keys(), cb.values(), label='bought')
        plt.legend()
        plt.show()


        plt.plot(pd.Series(all_rewards).rolling(window=20).mean(), label='Mean average precision @20')
        plt.legend()
        plt.title('moving ap@20 for each user')
        plt.show()

        plt.bar(counts_by_iid.keys(), counts_by_iid.values())
        plt.title('count of recommended items')
        plt.show()

        if n_users * n_items < 1000:
            plt.figure(figsize=(30,6))
            sns.heatmap(data=user_item_counts)
            plt.xlabel('item')
            plt.ylabel('user')
            plt.title('user-item count heatmap')
            plt.show()
        else:
            #print('Heatmap for %d is too big' % n_users*n_items)
            pass

        if hasattr(data[0], 'ranks'):

            ranks = []
            for i in data[:t]:
                ranks.append(i.ranks)
            ranks = np.array(ranks)

            print('Mean ranks for each position')
            print(np.mean(ranks, axis=0))

            plt.figure(figsize=(10, 5))
            for i in range(4):
                plt.plot(pd.Series(ranks[:, i]).rolling(window=30).mean(), label='%d' % (i + 1))
            plt.title('Moving average of ranks for each position')
            plt.legend()
            plt.show();

        else:
            print('No (ranks) info')


    interactive_plot = widgets.interactive(plot_counts, t=time_slider)
    return interactive_plot