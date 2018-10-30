import gym
import numpy as np


def generate_user():
    raise NotImplemented('gen user')
    return user_id, features


def generate_product():
    raise NotImplemented('gen product')
    return product_id, features


env_0_args = {
    'num_users' : 100,
    'num_products' : 200,
    'num_of_products_groups' : 5,
    'embedding_dimention' : 2,
    'user_preference_var' : 0.2,
}


class PrimEnv0(gym.Env):
    def __init__(self):
        """
        """
        self.users = {}
        self.products = {}
        self.history = []
        self.time = 0
        self.done = False

    def _transition(self):
        # users make organic events
        # add them to history

        # delete products with end_date >= self.time

        # add new products

        self.time += 1
        pass

    def _get_proba(self, action):
        pass

    def _get_reward(self, action):
        p_reward = self._get_proba(action)
        if np.random.rand() <= p_reward:
            reward = 1
            # user clicked
            # update self.history
            pass

        return reward


    def render(self, mode='human'):
        pass

    def step(self, action):
        """
        Parameters
        ----------
        action :  prod_id

        Returns
        -------
        observation, reward, done, info : tuple
            observation (tuple) :
                current_user : user_id
                    user that needs recommendations
                user_features (dict):
                    user_id : [ features ]
                product_features (dict):
                    prod_id : [ [features], begin_date, end_data ]
                purchases (tuple):
                    ( (user_id, prod_id, ts), ... )

            reward (float) :
                {0,1}
            done (bool) :
                recommendation session is done
            info (dict) :
                 this is unused, it's always an empty dict
        """
        reward = self._get_reward(action)

        self.transition()
        current_user = np.random.choice(self.users.keys())
        observation = (current_user, self.users, self.products, self.history)
        return observation, reward, self.done, None


    def reset(self):
        """
        regenerate users, products, initialize history
        :return:
        """
        self.users = {}
        self.products = {}
        self.history = []
        self.time = 0
        self.done = False

