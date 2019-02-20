import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, InputLayer, Activation, activations, Input, concatenate

from agents.utils import Agent
from agents.utils import ReplayBuffer

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Qagent(Agent):
    def __init__(self,
                 sess,
                 state_dim,
                 action_dim,
                 epsilon=0.4,
                 action_size=4,
                 logdir='./logs/',
                 replay_size=1000,
                 batch_size=64):

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_size = action_size

        self._logdir = logdir

        self._sess = sess

        self.epsilon = epsilon
        self.gamma = 0.9
        self.lr = 1e-4
        self.optimizer = tf.train.AdadeltaOptimizer(self.lr)

        self.state, self.action, self.agent, self.weights = self._create_network('agent')

        self.qvalues = self.agent(tf.concat([self.state, self.action], axis=-1))

        self.target_state, self.target_action, self.target, self.target_weights = self._create_network('target')
        self.target_qvalues = self.target(tf.concat([self.target_state, self.target_action], axis=-1))

        self.train_op, self.td_loss = self._create_train_op()
        self.target_update_op = self._create_target_update_op()

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self._logdir,
                                                  self._sess.graph)
        self.summary = None

        self._replay = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.td_losses = []

        self._last_state = None
        self._last_items = None
        self._last_action = None
        self.eval_mode = False
        self.training_steps = 0

    def _create_network(self, name):

        with tf.variable_scope(name_or_scope=name):
            state_ph = tf.placeholder('float32',
                                      shape=(None,) + self._state_dim,
                                      name='state')
            action_ph = tf.placeholder('float32',
                                       shape=(None,) + self._action_dim,
                                       name='action')

            net = Sequential(
                layers=[
                    InputLayer(input_shape=(self._state_dim[0] + self._action_dim[0],)),
                    Dense(200, activation='relu'),
                    Dense(100, activation='relu'),
                    Dense(1)
                ]
            )
            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

            return state_ph, action_ph, net, weights

    def _create_train_op(self):

        with tf.variable_scope(name_or_scope='train') as train_scope:

            # s a r s' A
            self.s_ph = tf.placeholder(tf.float32, shape=(None,) + self._state_dim, name='s')
            self.a_ph = tf.placeholder(tf.float32, shape=(None, self._action_size) + self._action_dim, name='a')
            self.r_ph = tf.placeholder(tf.float32, shape=[None], name='r')
            self.done_ph = tf.placeholder(tf.float32, shape=[None], name='done')
            self.next_s_ph = tf.placeholder(tf.float32, shape=(None,) + self._state_dim, name='next_s')
            # pool of actions at time T (ot T+1?)
            self.next_as_ph = tf.placeholder(tf.float32,
                                             shape=(None, None,) + self._action_dim, name='next_as')

            repeat_current_state = tf.expand_dims(self.s_ph, 1)
            repeat_current_state = tf.tile(repeat_current_state, multiples=[1, self._action_size, 1])
            current_qvalue = self.agent(tf.concat([repeat_current_state, self.a_ph], axis=-1))

            current_qvalue = tf.squeeze(current_qvalue, 0)
            current_qvalue = tf.reduce_sum(current_qvalue, axis=-1)

            repeat_states = tf.expand_dims(self.next_s_ph, 1)
            repeat_states = tf.tile(repeat_states, multiples=[1, tf.shape(self.next_as_ph)[1], 1])

            next_qvalues = self.target(tf.concat([repeat_states,
                                                  self.next_as_ph], axis=-1))
            next_qvalues = tf.squeeze(next_qvalues, axis=-1)

            k_max_next_qvalues, _ = tf.nn.top_k(next_qvalues, k=self._action_size)
            # should sum but not over batches
            next_max_qvalue = tf.reduce_sum(k_max_next_qvalues, axis=-1)

            reference = self.r_ph + self.gamma * next_max_qvalue

            td_loss = (current_qvalue - reference) ** 2
            td_loss = tf.reduce_mean(td_loss)

            tf.summary.histogram('next_max_qvalue', next_max_qvalue)
            tf.summary.histogram('topk', k_max_next_qvalues)
            tf.summary.histogram('target', reference)
            tf.summary.histogram('qvalue', current_qvalue)

            tf.summary.scalar('td_loss', td_loss)
            # Op to calculate every variable gradient
            grads = tf.gradients(td_loss, self.weights)
            grads = list(zip(grads, self.weights))
            # Summarize all gradients
            #for grad, var in grads:
            #    tf.summary.histogram(var.name + '/gradient', grad)

        return self.optimizer.minimize(td_loss, var_list=self.weights), td_loss

    def _create_target_update_op(self):
        """ assign target_network.weights variables to their respective agent.weights values. """
        assigns = []
        for w_agent, w_target in zip(self.weights, self.target_weights):
            assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
        return assigns

    def rank_action(self, state, actions):
        qvalues = self._sess.run(self.qvalues, {self.state: np.repeat(state.reshape(-1, self._state_dim[0]),
                                                                      actions.shape[0], axis=0),
                                                self.action: actions})
        return qvalues

    # add rank_actions target
    # action for environement is an array of items
    # action for q function is an item
    def target_rank_action(self, state, actions):
        return self._sess.run(self.target_qvalues, {
            self.target_state: np.repeat(state.reshape(-1, self._state_dim[0]),
                                         actions.shape[0], axis=0),
            self.target_action: actions,
        })

    def sample_action(self, state, actions):

        qvalues = self.rank_action(state, actions).reshape(-1)

        if np.random.rand() > self.epsilon:
            idxs = qvalues.argsort()[::-1][:self._action_size]
        else:
            idxs = np.random.choice(range(actions.shape[0]), size=self._action_size,
                                    p=softmax(qvalues))

        return actions[idxs]

    def _train(self, batch):
        s, a, r, next_s, actions, done = batch

        losses = []
        for s, a, r, next_s, actions, done in zip(*batch):

            _, loss, _ = self._sess.run([self.train_op, self.td_loss, self.merged], {
                self.s_ph: s[None],
                self.a_ph: a[None],
                self.r_ph: r[None],
                self.done_ph: done[None],
                self.next_s_ph: next_s[None],
                self.next_as_ph: actions[None]
            })
            losses.append(loss)

        return np.mean(losses)

    def update_target_weights(self):
        self._sess.run(self.target_update_op)

    def write_summary(self, step):
        if self.summary:
            self.train_writer.add_summary(self.summary, global_step=step)

    def _sample_action(self, state, actions):
        actions = np.array(actions)

        qvalues = self.rank_action(state, actions).reshape(-1)

        if np.random.rand() > self.epsilon:
            idxs = qvalues.argsort()[::-1][:self._action_size]
        else:
            idxs = np.random.choice(range(actions.shape[0]), size=self._action_size,
                                    p=softmax(qvalues))
        return idxs, actions[idxs]

    def _train_step(self):
        if len(self._replay) >= self.batch_size:
            self.training_steps += 1

            batch = self._replay.sample(self.batch_size)
            td_loss = self._train(batch)
            self.td_losses.append(td_loss)

    def begin_episode(self, observation):
        state, items = observation
        self._last_state = state
        self._last_items = items
        actions_ids, action = self._sample_action(state, items)
        self._last_action = action
        return actions_ids

    def step(self, reward, observation):
        state, items = observation
        self._replay.add(self._last_state, self._last_action, reward, state, np.array(items), False)

        if not self.eval_mode:
            self._train_step()

        self._last_state = state
        self._last_items = items
        actions_ids, action = self._sample_action(state, items)
        self._last_action = action
        return actions_ids

    def end_episode(self, reward):
        return super().end_episode(reward)

    def bundle_and_checkpoint(self, directory, iteration):
        return super().bundle_and_checkpoint(directory, iteration)

    def unbundle(self, directory, iteration, dictionary):
        return super().unbundle(directory, iteration, dictionary)
