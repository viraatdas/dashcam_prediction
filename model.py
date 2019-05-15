#Based off https://github.com/JovanSardinha/speed-challenge-2017

import cv2
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


seeds = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def train_valid_split(dframe, seed_val):
    """

    :param dframe:
    :param seed_val:
    :return: tuple(train_data, valid_data) df

    Shuffle pairs of rows in the df, and seperates train and validation data
    """

    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()

    np.random.seed(seed_val)
    for i in tqdm(range(len(dframe)) - 1):
        idx1 = np.random.randint(len(dframe) - 1)
        idx2 = idx1 + 1

        row1 = dframe.iloc[[idx1]].reset_index()
        row2 = dframe.iloc[[idx2]].reset_index()

        randInt = np.random.randint(9)
        if 0 <= randInt <= 1:
            valid_frames = [valid_data, row1, row2]
            valid_data = pd.concat(valid_frames, axis = 0, join = 'outer', ignore_index=False)
        if randInt >= 2:
            train_frames = [train_data, row1, row2]
            train_data = pd.concat(train_frames, axis = 0, join = 'outer', ignore_index=False)
        return train_data, valid_data
train_meta = pd.read_csv("data/img_csv/train_info.csv")
train_data, valid_data = train_valid_split(train_meta, seeds[0])

fig, ax = plt.subplots(figsize=(20,10))
plt.plot(train_data.sort_values(['image_index'])[['image_index']], train_data.sort_values(['image_index'])[['speed']], 'ro')
plt.plot(valid_data.sort_values(['image_index'])[['image_index']], valid_data.sort_values(['image_index'])[['speed']], 'go')
plt.xlabel('image_index (or time since start)')
plt.ylabel('speed')
plt.title('Speed vs time')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./assets/speed_vs_time_val_train.png')
plt.close()

print('----')
print('valid_data: ', valid_data.shape)
print('train_data: ', train_data.shape)




def opticalFlowDense(curr_img, next_img):
    gray_current = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros((66, 220, 3))

    #set saturation
    hsv[:,:,1] = cv2.cvtCOLOR(next_img, cv2.COLOR_RGB2HSV)[:,:,1]

    #Flow Parameters
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    #dense optical flow parameters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,
                                        flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        0)

    #convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    #hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)

    #value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    #convert HSV to int32's
    hsv = np.asarray(hsv, dtype = np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow


