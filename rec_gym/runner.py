# import numpy as np

# def play_and_record(agent, env, exp_replay, n_steps=1, greedy=False):
#     # initial state
#     s = env.reset()
#
#     rewards = []
#     # Play the game for n_steps as per instructions above
#     for _ in range(n_steps):
#
#         (user, items) = s
#
#         action_ids, action = agent.sample_action(user, items)
#
#         next_s, r, done, _ = env.step(action_ids)
#
#         rewards.append(r)
#
#         (next_user, next_items) = next_s
#
#         exp_replay.add(user, action, r,
#                        next_user, next_items, done)
#
#         if done:
#             s = env.reset()
#         else:
#             s = next_s
#
#     return np.mean(rewards)


# def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
#     """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
#     rewards = []
#
#     for _ in range(n_games):
#         (user, items) = env.reset()
#
#         s = user
#         actions = items
#
#         reward = 0
#         for _ in range(t_max):
#
#             action_ids, action = agent.sample_action(s, actions)
#
#             obs, r, done, _ = env.step(action_ids)
#
#             (next_user, next_items) = obs
#
#             s = next_user
#             actions = next_items
#
#             reward += r
#             if done: break
#
#         rewards.append(reward)
#     return np.mean(rewards)

from tqdm import tqdm

def run_experiment(env, agent, t_train=10000, t_test=2000):
    observation = env.reset()
    action = agent.begin_episode(observation)

    for i in tqdm(range(t_train)):
        observation, reward, done, info = env.step(action)
        action = agent.step(reward, observation)

    for i in tqdm(range(t_test)):
        observation, reward, done, info = env.step(action)
        action = agent.step(reward, observation)