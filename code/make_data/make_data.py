#%% Environment
from copy import copy
import os
from os.path import dirname
from shutil import copyfile
import shutil
import random


#%% Functions
def split_data(source_path, training_path, validation_path, training_ratio=0.9):
    """
    
    """
    print('== Begin to split data ==')
    print(f'Source path is {source_path}')
    print(f'training path is {training_path}')
    print(f'validation path is {validation_path}')
    
    # make training and validation path
    if os.path.exists(training_path):
        shutil.rmtree(training_path)
    if os.path.exists(validation_path):
        shutil.rmtree(validation_path)
    
    os.makedirs(training_path)
    os.makedirs(validation_path)
    
    file_list = list()
    for i in os.listdir(source_path):
        if os.path.getsize(os.path.join(source_path, i)) > 0:
            file_list.append(i)
        else:
            print(f'ignore file {i} for it\'s size is 0,')
    
    tr_files = random.sample(file_list, int(training_ratio * len(file_list)))
    
    for i in file_list:
        if i in tr_files:
            des_path = os.path.join(training_path, i)
        else:
            des_path = os.path.join(validation_path, i)
        copyfile(os.path.join(source_path, i), des_path)

    print('== Finish spliting data ==')

#%% Operation
if __name__=='__main__':
    # root path
    script_path = os.path.abspath(__file__)
    root_path = dirname(dirname(dirname(script_path)))
    os.chdir(root_path)
    # make training and validation path
    tr_cat_path = os.path.join(root_path, 'data', 'training', 'cat')
    tr_dog_path = os.path.join(root_path, 'data', 'training', 'dog')   
    val_cat_path = os.path.join(root_path, 'data', 'validation', 'cat')   
    val_dog_path = os.path.join(root_path, 'data', 'validation', 'dog')   
    
    # source_path
    source_cat = os.path.join(root_path, 'assets', 'data', 'Cat')
    source_dog = os.path.join(root_path, 'assets', 'data', 'Dog')
    
    split_data(source_cat, tr_cat_path, val_cat_path)
    split_data(source_dog, tr_dog_path, val_dog_path)
    