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