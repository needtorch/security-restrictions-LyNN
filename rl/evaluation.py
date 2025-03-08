from typing import Dict

import gymnasium as gym
import numpy as np


def evaluate(agent, env: gym.Env, num_episodes: int, tseed) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for _ in range(num_episodes):
        observation, info = env.reset(seed=(tseed+42))
        terminated = False
        truncated = False
        while (not truncated) and (not terminated):
            action = agent.eval_actions(observation)
            observation, _, terminated, truncated, info = env.step(action)
            # next_observation, reward, terminated, truncated, info

    return {
        'return': np.mean(env.return_queue),
        'length': np.mean(env.length_queue)
    }
