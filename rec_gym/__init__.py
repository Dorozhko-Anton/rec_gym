from .envs import env_0_args, env_1_args
from gym.envs.registration import register

register(
    id='prim-gym-v0',
    entry_point='rec_gym.envs:PrimEnv0'
)

register(
    id='prim-gym-v1',
    entry_point='rec_gym.envs:PrimEnv1'
)