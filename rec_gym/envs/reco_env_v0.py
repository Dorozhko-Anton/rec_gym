import gym
import numpy as np

def purchase_proba(user, product, ts):
    raise NotImplemented('purchase_proba')


def generate_user(seed):

    user = {
        'age' : ,
        'sex' : ,
        'mean_expe'
    }

    raise NotImplemented('gen user')

def generate_product(seed):
    raise NotImplemented('gen product')



env_0_args = {
    'num_users' : 100,
    'num_products' : 200,

}




class PrimEnv0(gym.Env):
    def __init__(self):
        """

        """
        pass

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
                user_features (dict):
                    user_id : [ features ]
                product_features (dict):
                    prod_id : [ [features], begin_date, end_data ]
                purchases (tuple):
                    ( (user_id, prod_id, ts), ... )

            reward (float) :

            done (bool) :

            info (dict) :
                 this is unused, it's always an empty dict
        """


        pass

    def reset(self):
        """

        :return:
        """
        pass