#!/usr/bin/env python3
import gym
import keras
from rl.policy import GreedyQPolicy
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

create_q_model = __import__('train').create_q_model
AtariProcessor = __import__('train').AtariProcessor

if __name__ == '__main__':
    env = gym.make('Breakout-v4')
    env.reset()
    window = 4
    actions_n = env.action_space.n
    model = create_q_model(actions_n, window)
    memory = SequentialMemory(limit=1000000, window_length=window)

    policy = GreedyQPolicy()
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

    agent.load_weights('policy.h5')

    agent.test(env, nb_episodes=10, visualize=True)