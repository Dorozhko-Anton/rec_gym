# from .envs import env_1_args
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
        'embedding_dimension': 20,
        'n_items_to_recommend': 4,
        'seed': 0,
        'n_users': 40,
        'n_items': 500,
        'normalize_reward': False
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
        'embedding_dimension': 20,
        'n_items_to_recommend': 1,
        'seed': 0,
        'n_users': 40,
        'n_items': 500,
        'normalize_reward': False
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

register(
    id='GenRecEnv-v1',
    entry_point='rec_gym.envs:GeneratedRecEnv',
    kwargs={
        'n_items': 400,
        'n_users': 100,
        'n_rec': 4,
        'embedding_dimension': 2,
        'user_change_prob': 0.05,
        'reward_noise': 0.05,
        'user_initial_n_clusters': 4,
        'user_init_sigma': 4,
        'user_cluster_sigma': 1,
        'user_ar_coef': 0.9,
        'user_drift_sigma': 0.3,
        'initial_n_clusters': 10,
        'cluster_var': 4,
        'in_cluster_var': 1,
        'new_items_interval': 1000,
        'new_items_size': 0,
        'click_prob_type': 'normal',
        'user_preference_type': 'static',
        'choose_only_one_item': False
    }
)


register(
    id='GenRecEnv-v2',
    entry_point='rec_gym.envs:GeneratedRecEnv',
    kwargs={
        'n_items': 400,
        'n_users': 100,
        'n_rec': 1,
        'embedding_dimension': 2,
        'user_change_prob': 0.05,
        'reward_noise': 0.05,
        'user_initial_n_clusters': 4,
        'user_init_sigma': 4,
        'user_cluster_sigma': 1,
        'user_ar_coef': 0.9,
        'user_drift_sigma': 0.3,
        'initial_n_clusters': 10,
        'cluster_var': 4,
        'in_cluster_var': 1,
        'new_items_interval': 1000,
        'new_items_size': 0,
        'click_prob_type': 'normal',
        'user_preference_type': 'static',
        'choose_only_one_item': False
    }
)

register(
    id='GenRecEnv-v3',
    entry_point='rec_gym.envs:GeneratedRecEnv',
    kwargs={
        'n_items': 400,
        'n_users': 10,
        'n_rec': 1,
        'embedding_dimension': 2,
        'user_change_prob': 0.05,
        'reward_noise': 0.05,
        'user_initial_n_clusters': 4,
        'user_init_sigma': 4,
        'user_cluster_sigma': 1,
        'user_ar_coef': 0.9,
        'user_drift_sigma': 0.3,
        'initial_n_clusters': 10,
        'cluster_var': 4,
        'in_cluster_var': 1,
        'new_items_interval': 1000,
        'new_items_size': 0,
        'click_prob_type': 'normal',
        'user_preference_type': 'static',
        'choose_only_one_item': False
    }
)


register(
    id='JDSimulatedEnv-v1',
    entry_point='rec_gym.envs:JDSimulatedEnv',
    kwargs={
        'n_users':20,
        'n_items':100,
        'embedding_size':2,
        'recommendation_size':4,
        'kappa': 0.5,
        'diversity_engagement_type':'linear',
        'a':1,
        'b':0,
        'V':1,
        'd':1,
        'mu':0,
        'sigma':1,
        'seed':None,
    }
)


register(
    id='JDSimulatedEnv-v2',
    entry_point='rec_gym.envs:JDSimulatedEnv',
    kwargs={
        'n_users':20,
        'n_items':100,
        'embedding_size':2,
        'recommendation_size':4,
        'kappa': 0.5,
        'diversity_engagement_type':'quadratic',
        'a':1,
        'b':0,
        'V':1,
        'd':1,
        'mu':0,
        'sigma':1,
        'seed':None,
    }
)