#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
import numpy as np

np.random.seed(0)
env = load_frozen_lake()
print(env.desc)
print(env.P[2][2])
env = load_frozen_lake(is_slippery=True)
print(env.desc)
print(env.P[2][2])
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
print(env.desc)
print(env.P[2][0])
print(env.P[2][2])
env = load_frozen_lake(map_name='4x4')
print(env.desc)
print(env.P[0][2])
print(env.P[2][2])
env = load_frozen_lake(is_slippery=True)
print(env.desc)
print(env.P[3][3])
print(env.P[3][2])
print(env.desc.shape)
print(sqrt(8))
