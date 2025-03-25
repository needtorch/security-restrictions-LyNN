from typing import Dict

import gymnasium as gym
import numpy as np
import pickle

def evaluate(agent, env: gym.Env, num_episodes: int, tseed) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    velocity = []
    term = []
    cost = []
    action_test = []
    observation_test = []
    for _ in range(num_episodes):
        observation, info = env.reset(seed=(tseed+42))
        observation_test.append(observation)
        terminated = False
        truncated = False
        while (not truncated) and (not terminated):
            action = agent.eval_actions(observation)
            observation, _, terminated, truncated, info = env.step(action)
            action_test.append(action)
            observation_test.append(observation)
            cost.append(info['cost'])
            term.append(terminated)
            velocity.append(info['x_velocity'])
            # next_observation, reward, terminated, truncated, info
    with open('./ant_eval/terminated_1k_' + str(tseed) + '.pkl', 'wb') as f:
        pickle.dump(term, f)
    with open('./ant_eval/cost_1k_' + str(tseed) + '.pkl', 'wb') as f:
        pickle.dump(cost, f)
    with open('./ant_eval/action_1k_' + str(tseed) + '.pkl', 'wb') as f:
        pickle.dump(action_test, f)
    with open('./ant_eval/observation_1k_' + str(tseed) + '.pkl', 'wb') as f:
        pickle.dump(observation_test, f)
    with open('./ant_eval/velocity_test_1k_' + str(tseed) + '.pkl', 'wb') as f:
        pickle.dump(velocity, f)

    return {
        'return': np.mean(env.return_queue),
        'length': np.mean(env.length_queue),
        'velocity': np.mean(velocity),
        'cost': sum(cost),
        'term': sum(term),
    }
