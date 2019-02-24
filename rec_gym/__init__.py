#from .envs import env_1_args
from gym.envs.registration import register

# register(
#     id='prim-gym-v0',
#     entry_point='rec_gym.envs:PrimEnv0'
# )

register(
    id='prim-gym-v1',
    entry_point='rec_gym.envs:PrimEnv1'
)

register(
    id='prim-gym-v2',
    entry_point='rec_gym.envs:PrimEnv2'
)


register(
    id='prim-gym-v3',
    entry_point='rec_gym.envs:PrimEnv3'
)

register(
    id='prim-gym-v2-ref-v1',
    entry_point='rec_gym.envs:PrimEnv2Ref'
)


register(
    id='MovieLens-v1',
    entry_point='rec_gym.envs:MovieLens',
    kwargs={
            'embedding_dimension':20,
            'n_items_to_recommend':4,
            'seed':0,
            'n_users':40,
            'n_items':500,
            'normalize_reward':False
    }
)

register(
    id='MovieLens-v2',
    entry_point='rec_gym.envs:MovieLens',
    kwargs={
        'embedding_dimension': 20,
        'n_items_to_recommend': 4,
        'seed': 0,
        'n_users': 40,
        'n_items': 500,
        'normalize_reward': True
    }
)


register(
    id='MovieLens-v3',
    entry_point='rec_gym.envs:MovieLens',
    kwargs={
            'embedding_dimension':20,
            'n_items_to_recommend':1,
            'seed':0,
            'n_users':40,
            'n_items':500,
            'normalize_reward':False
    }
)

register(
    id='MovieLens-v4',
    entry_point='rec_gym.envs:MovieLens',
    kwargs={
        'embedding_dimension': 20,
        'n_items_to_recommend': 1,
        'seed': 0,
        'n_users': 40,
        'n_items': 500,
        'normalize_reward': True
    }
)