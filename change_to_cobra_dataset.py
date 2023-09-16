import os
import random
import tqdm
import numpy as np
from PIL import Image
from datetime import datetime
import yaml
import h5py
import gym
import elastic2d
import cv2
from sklearn.preprocessing import OneHotEncoder

dirname = "results/orn/"
dataset_name = "rlb_5_33_c"

def resize(img):
    return cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_LINEAR)

def open_image_in(idx):
    return np.array(Image.open(dirname + dataset_name + "/" + "train/input/" + dataset_name + f"_{idx}.png"))

def open_image_out(idx):
    return np.array(Image.open(dirname + dataset_name + "/" + "train/output/" + dataset_name + f"_{idx}.png"))


features = np.zeros((21, 9900, 64, 64, 3))
features_test = np.zeros((21, 100, 64, 64, 3))
actions = np.zeros((21, 9900, 2))
actions_test = np.zeros((21, 100, 2))


action_load = np.loadtxt(dirname + dataset_name + f"/train/{dataset_name}_actions.csv", delimiter=',', dtype=np.int32)
# action_load_one_hot = np.eye(2)[action_load]

# train
for i in tqdm.tqdm(range(198000)):
    # features[i%20][i//20] = resize(open_image_in(i - i%20))
    if i % 20 == 0:
        features[0][i//20] = resize(open_image_in(i - i%20))    
        features[1][i//20] = resize(open_image_out(i))
    else:
        features[i%20 + 1][i//20] = resize(open_image_out(i))
    actions[i%20][i//20] = action_load[i]


# test
for i in range(198000,200000):
    # features[i%20][(i-198000)//20] = resize(open_image_in(i - i%20))
    if i % 20 == 0:
        features_test[0][(i-198000)//20] = resize(open_image_in(i))
        features_test[1][(i-198000)//20] = resize(open_image_out(i))
    else:
        features_test[i%20 + 1][(i-198000)//20] = resize(open_image_out(i))
    actions_test[i%20][(i-198000)//20] = action_load[i]
    


with h5py.File(f'elastic_{dataset_name}_data.h5', 'w') as f:
    train_folder = f.create_group("training")
    train_image_data_shape = features.shape
    train_action_data_shape = actions.shape
    train_image_dataset = train_folder.create_dataset('features', train_image_data_shape, dtype='uint8')
    train_action_dataset = train_folder.create_dataset('actions', train_action_data_shape, dtype='float32')

    test_folder = f.create_group("validation")
    test_image_data_shape = features_test.shape
    test_action_data_shape = actions_test.shape
    test_image_dataset = test_folder.create_dataset('features', test_image_data_shape, dtype='uint8')
    test_action_dataset = test_folder.create_dataset('actions', test_action_data_shape, dtype='float32')

    train_image_dataset[:, :, :, :, :] = features
    train_action_dataset[:, :, :] = actions

    test_image_dataset[:, :, :, :, :] = features_test
    test_action_dataset[:, :, :] = actions_test
