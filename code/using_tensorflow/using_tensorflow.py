#%% Environment
# package
import tensorflow as tf
import tensorflow.keras as keras
import os
from os.path import dirname

# root path
script_path = os.path.abspath(__file__)
root_path = dirname(dirname(dirname(script_path)))
os.chdir(root_path)

# data path
train_path = os.path.join(root_path, 'data', 'training')

#%% Data
# DataLoader with tensorflow
