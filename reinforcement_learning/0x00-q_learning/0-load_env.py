#!/usr/bin/env python3
"""loads the pre-made FrozenLakeEnv
evnironment from OpenAI’s gym"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Returns: the environment"""
    if desc is not None and map_name is not None:
        raise ValueError("Only one of 'desc' or 'map_name' should be provided.")

    if map_name is not None:
        env = gym.make('FrozenLake-v1', map_name=map_name)
    else:
        env = gym.make('FrozenLake-v1', desc=desc, is_slippery=is_slippery, render_mode='ansi')

    return env
