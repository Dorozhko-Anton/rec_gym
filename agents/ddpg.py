import tensorflow as tf
import gin

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate

import numpy as np
import os

from agents.utils import ReplayBuffer, Agent


@gin.configurable
class ActorNetwork:
    def __init__(self,
                 action_size: int,
                 state_dim: int,
                 action_dim: int,
                 sess: tf.Session,
                 optimizer: tf.train.Optimizer = tf.train.AdamOptimizer(
                     learning_rate=0.001
                 )):

        self.optimizer = optimizer
        self.sess = sess
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_size = action_size

        self._actor_gradients_ph = tf.placeholder(tf.float32,
                                                  shape=[None, self.action_dim * self.action_size],
                                                  name='action_gradients')
        self._state_ph = tf.placeholder(tf.float32,
                                        shape=[None, self.state_dim],
                                        name='state')
        self._items_ph = tf.placeholder(tf.float32,
                                        shape=[None, self.action_dim],
                                        name='items')

        self._actor = self._actor_template('actor_online')
        self._actor_target = self._actor_template('actor_target')

        self._actor_weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_online')

        self.scores_op = tf.matmul(a=tf.reshape(self._actor,
                                                shape=[self.action_size, self.action_dim]),
                                   b=self._items_ph, transpose_b=True)

        self._sync_op = self._build_actor_sync_op()
        self._train_op = self._build_train_op()

    def predict_action(self, state, items):

        scores = self.sess.run(self.scores_op, feed_dict={self._state_ph: [state],
                                                          self._items_ph: items})

        actions_ids = []
        for i in range(self.action_size):
            for chosen in actions_ids:
                scores[i][chosen] = 0

            # actions_ids.append(np.argmax(scores[i]))
            def normalize(x):
                return (x-np.min(x))/np.sum((x-np.min(x)))
            actions_ids.append(np.random.choice(range(len(scores[i])), p=normalize(scores[i]) ))

        return actions_ids, np.array(items)[actions_ids]

    def train(self, state, action_gradients):
        self.sess.run(self._train_op, feed_dict={
            self._state_ph: state,
            self._actor_gradients_ph: action_gradients
        })

    def sync_target(self):
        self.sess.run(self._sync_op)

    # utils functions
    def _build_train_op(self):

        gradients = tf.gradients(self._actor, self._actor_weights, grad_ys=-self._actor_gradients_ph)
        actor_gradients = list(map(lambda x: tf.div(x, tf.cast(tf.shape(self._state_ph)[0], tf.float32)),
                                   gradients))

        # Optimization Op
        optimize = self.optimizer.apply_gradients(zip(actor_gradients, self._actor_weights))
        return optimize

    def _actor_template(self, scope):
        """Builds the actor network used to compute the agent's scores for items.
        Returns:
        """
        with tf.variable_scope(name_or_scope=scope):
            net = Dense(200, activation='relu')(self._state_ph)
            net = Dense(100, activation='relu')(net)
            net = Dense(self.action_size * self.action_dim)(net)

        return net

    def _build_actor_sync_op(self):
        """Builds ops for assigning weights from online to target network.
        Returns:
          ops: A list of ops assigning weights from online to target network.
        """
        # Get trainable variables from online and target DQNs
        sync_qt_ops = []
        trainables_online = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_online')
        trainables_target = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target')
        for (w_online, w_target) in zip(trainables_online, trainables_target):
            # Assign weights from online to target network.
            sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
        return sync_qt_ops


@gin.configurable
class CriticNetwork:
    def __init__(self,
                 action_size: int,
                 state_dim: int,
                 action_dim: int,
                 gamma: float,
                 sess: tf.Session,
                 optimizer: tf.train.Optimizer = tf.train.AdamOptimizer(
                     learning_rate=0.001
                 ),
                 ):
        self.gamma = gamma
        self.optimizer = optimizer
        self.sess = sess
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_size = action_size

        self._state_ph = tf.placeholder(tf.float32,
                                        shape=[None, self.state_dim],
                                        name='state')
        self._action_ph = tf.placeholder(tf.float32,
                                         shape=[None, self.action_dim * self.action_size],
                                         name='action')
        self._reference_ph = tf.placeholder(tf.float32,
                                            shape=[None, 1],
                                            name='qvalue_reference')

        self._critic = self._critic_template('critic_online')
        self._critic_target = self._critic_template('critic_target')
        self._critic_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope='critic_online')

        self.sync_op = self._build_critic_sync_op()
        self._build_train_op()

    def sync_target(self):
        self.sess.run(self.sync_op)

    def predict_qvalue(self, state, action):
        return self.sess.run(self._critic, feed_dict={
            self._state_ph: state,
            self._action_ph: action,
        })

    def predict_target_qvalue(self, state, action):
        return self.sess.run(self._critic_target, feed_dict={
            self._state_ph: state,
            self._action_ph: action,
        })

    def train(self, batch):
        s, a, r, s_next, a_next = batch

        reference_values = self.predict_target_qvalue(s_next, a_next)

        _, td_loss, action_gradients = self.sess.run([self.train_op, self.td_loss, self.action_gradients],
                                                feed_dict={
                                                    self._state_ph: s,
                                                    self._action_ph: a,
                                                    self._reference_ph: reference_values,
                                                })
        return td_loss, action_gradients

    # utils
    def _critic_template(self, scope):
        """Builds the critic network used to compute the agent's scores for (s, a).
        Returns:
            net
        """
        with tf.variable_scope(name_or_scope=scope):
            net = concatenate([self._state_ph, self._action_ph], axis=-1)
            net = Dense(50, activation='relu')(net)
            net = Dense(30, activation='relu')(net)
            net = Dense(1)(net)

        return net

    def _build_critic_sync_op(self):
        """Builds ops for assigning weights from online to target network.
        Returns:
          ops: A list of ops assigning weights from online to target network.
        """
        # Get trainable variables from online and target DQNs
        sync_qt_ops = []
        trainables_online = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_online')
        trainables_target = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_target')
        for (w_online, w_target) in zip(trainables_online, trainables_target):
            # Assign weights from online to target network.
            sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
        return sync_qt_ops

    def _build_train_op(self):
        self.td_loss = (self._critic - self.gamma * self._reference_ph) ** 2
        self.td_loss = tf.reduce_mean(self.td_loss)
        # Optimization Op
        self.train_op = self.optimizer.minimize(self.td_loss)
        self.action_gradients = tf.gradients(self.td_loss,
                                             self._action_ph, name='action_gradients')[0]



@gin.configurable
class DDPGAgent(Agent):
    def __init__(self,
                 action_size: int,
                 state_dim: int,
                 action_dim: int,
                 gamma: float,
                 sess: tf.Session,
                 optimizer: tf.train.Optimizer = tf.train.AdamOptimizer(
                     learning_rate=0.001
                 ),
                 max_tf_checkpoints_to_keep: int = 3,
                 experience_size: int = 1000,
                 batch_size: int = 64
                ):
        self.optimizer = optimizer
        self.sess = sess
        self.gamma = gamma
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_size = action_size

        self.actor = ActorNetwork(action_size=action_size, state_dim=state_dim,
                                  action_dim=action_dim, sess=sess, optimizer=optimizer)

        self.critic = CriticNetwork(action_size=action_size, state_dim=state_dim,
                                    action_dim=action_dim, sess=sess, optimizer=optimizer, gamma=gamma)

        self.eval_mode = False
        self.training_steps = 0
        self.epsilon = 1
        self.batch_size = batch_size

        self._saver = tf.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)
        self._replay = ReplayBuffer(experience_size)

        self._last_state = None
        self._last_items = None
        self._last_action = None

        self.td_losses = []

    def begin_episode(self, observation):
        state, items = observation
        self._last_state = state
        self._last_items = items
        actions_ids, action = self._sample_action(state, items)
        self._last_action = action
        return actions_ids

    def step(self, reward, observation):
        state, items = observation

        self._replay.add(self._last_state, self._last_action, reward, state, items, False)

        if not self.eval_mode:
            self._train_step()

        self._last_state = state
        self._last_items = items
        actions_ids, action = self._sample_action(state, items)
        self._last_action = action
        return actions_ids

    def end_episode(self, reward):
        return super().end_episode(reward)

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Returns a self-contained bundle of the agent's state.

        This is used for checkpointing. It will return a dictionary containing all
        non-TensorFlow objects (to be saved into a file by the caller), and it saves
        all TensorFlow objects into a checkpoint file.

        Args:
          checkpoint_dir: str, directory where TensorFlow objects will be saved.
          iteration_number: int, iteration number to use for naming the checkpoint
            file.

        Returns:
          A dict containing additional Python objects to be checkpointed by the
            experiment. If the checkpoint directory does not exist, returns None.
        """
        if not tf.gfile.Exists(checkpoint_dir):
            return None
        # Call the Tensorflow saver to checkpoint the graph.
        self._saver.save(
            self._sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration_number)
        # Checkpoint the out-of-graph replay buffer.
        #self._replay.save(checkpoint_dir, iteration_number)
        bundle_dictionary = {}
        #bundle_dictionary['state'] = self.state
        #bundle_dictionary['eval_mode'] = self.eval_mode
        #bundle_dictionary['training_steps'] = self.training_steps
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        """Restores the agent from a checkpoint.

        Restores the agent's Python objects to those specified in bundle_dictionary,
        and restores the TensorFlow objects to those specified in the
        checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
          agent's state.

        Args:
          checkpoint_dir: str, path to the checkpoint saved by tf.Save.
          iteration_number: int, checkpoint version, used when restoring replay
            buffer.
          bundle_dictionary: dict, containing additional Python objects owned by
            the agent.

        Returns:
          bool, True if unbundling was successful.
        """
        try:
            pass
            # self._replay.load() will throw a NotFoundError if it does not find all
            # the necessary files, in which case we abort the process & return False.
            #self._replay.load(checkpoint_dir, iteration_number)
        except tf.errors.NotFoundError:
            return False
        for key in self.__dict__:
            if key in bundle_dictionary:
                self.__dict__[key] = bundle_dictionary[key]
        # Restore the agent's TensorFlow graph.
        self._saver.restore(self._sess,
                            os.path.join(checkpoint_dir,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True

    def _train_step(self):
        if len(self._replay) >= self.batch_size:
            self.training_steps += 1

            batch = self._replay.sample(self.batch_size)
            td_loss = self._train(batch)
            self.td_losses.append(td_loss)

    def _train(self, batch):
        state, action, r, next_s, items, done = batch
        action = [np.reshape(a, newshape=-1) for a in action]
        # choose actions for next_s
        a_next = []
        #print("_train")
        #print(state.shape)
        #print(state[0])
        #print("="*20)
        for i in range(len(state)):
            #print(state[i])
            ids, next_action = self.actor.predict_action(state=state[i], items=items[i])
            a_next.append(np.reshape(next_action, newshape=-1))
        td_loss, action_gradients = self.critic.train([state, action, r, next_s, a_next])
        self.actor.train(state=state, action_gradients=action_gradients)
        return td_loss

    def _sample_action(self, observation, items):
        if len(self._replay) < 2*self.batch_size:
            actions_ids = np.random.choice(range(len(items)), size=self.action_size)
            return actions_ids, [items[i] for i in actions_ids]

        actions_ids, action = self.actor.predict_action(observation, items)
        return actions_ids, action

    def _update_target_weights(self):
        self.actor.sync_target()
        self.critic.sync_target()


if __name__ == '__main__':
    with tf.Session() as sess:
        actor = ActorNetwork(action_size=2, state_dim=2, action_dim=2, sess=sess)
        critic = CriticNetwork(action_size=2, state_dim=2, action_dim=2, sess=sess, gamma=0.9)

        sess.run(tf.global_variables_initializer())

        state = [[1, 1],
                 [1, 1]]

        items = [
            [[1, 0],
             [0, 1]],

            [[1, 0],
             [0, 1],
             [1, 1]],
        ]
        action_gradients = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

        # test
        actions_ids, action = actor.predict_action(state=state[0], items=items[0])
        print(actions_ids, action)
        actor.train(state=state, action_gradients=action_gradients)

        #
        a_next = []
        for i in range(len(state)):
            ids, action = actor.predict_action(state=state[i], items=items[i])

            a_next.append(np.reshape(action, newshape=-1))

        print(a_next)
        qvalue = critic.predict_qvalue(state, a_next)
        print(qvalue)

        r = np.ones(len(state))

        for i in range(10):
            print(i)
            td_loss, action_gradients = critic.train([state, a_next, r, state, a_next])
            print("td_loss = %s\n a_grads=%s" % (td_loss, action_gradients))

            actor.train(state=state, action_gradients=action_gradients)
