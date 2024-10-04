#!/usr/bin/env python3
"""train.py"""
from PIL import Image
import gym
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor


class AtariProcessor(Processor):
    """Preprocessing Images"""
    def process_observation(self, observation):
        """Converts to gray an rescales observation"""
        return np.array(Image.fromarray(observation).convert('L').resize((84, 84)))


    def process_state_batch(self, batch):
        """Rescale the batch of images"""
        image_n = batch.astype('float32') / 255.
        return image_n

    def process_reward(self, reward):
        """-1>rewards< 1"""
        return np.clip(reward, -1., 1.)


def create_q_model(actions, window):
    """create Sequential model for agent"""
    model = Sequential()
    # add one layer at a time in sequence
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(window, 84, 84)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


if __name__ == '__main__':
    """mnjm"""
    env = gym.make("Breakout-v4")
    np.random.seed(0)
    env.reset()

    window = 4
    actions_n = env.action_space.n
    model = create_q_model(actions_n, window)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.05,
        nb_steps=1000000
    )

    memory = SequentialMemory(
        limit=1000000,
        window_length=window
    )

    agent = DQNAgent(
        model=model,
        nb_actions=actions_n,
        policy=policy,
        memory=memory,
        processor=AtariProcessor(),
        nb_steps_warmup=50000,
        target_model_update=10000,
        gamma=.99
    )

    agent.compile(keras.optimizers.Adam(), metrics=['mae'])

    agent.fit(
        env,
        nb_steps=1750000,
        visualize=False,
        verbose=2
    )

    agent.save_weights('policy.h5', overwrite=True)