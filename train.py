from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
from PIL import Image
import time
import argparse
import pyflow
import pickle
import multiprocessing
from multiprocessing import Pool
import numpy as np 
import pandas as pd
import os
import cv2
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import datetime
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import UpSampling2D, ZeroPadding2D, concatenate, Flatten
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, Conv2DTranspose, Dense
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras import backend as K



dataset=[i for i in range(0,29)]
frames=[]

def k_model(shape):

    x = Input(shape=shape, )

    conv0 = Conv2D(64,(3,3),padding='same',name='conv0')(x)
    conv0 = LeakyReLU(0.1)(conv0)
    padding = ZeroPadding2D()(conv0)

    conv1 = Conv2D(64,(3,3),strides=(2,2),padding='valid',name='conv1')(padding)
    conv1 = LeakyReLU(0.1)(conv1)
    conv1_1 = Conv2D(128,(3,3),padding='same',name='conv1_1')(conv1)
    conv1_1 = LeakyReLU(0.1)(conv1_1)
    padding = ZeroPadding2D()(conv1_1)

    conv2 = Conv2D(128,(3,3),strides=(2,2),padding='valid',name='conv2')(padding)
    conv2 = LeakyReLU(0.1)(conv2)
    conv2_1 = Conv2D(128,(3,3),padding='same',name='conv2_1')(conv2)
    conv2_1 = LeakyReLU(0.1)(conv2_1)
    padding = ZeroPadding2D()(conv2_1)

    conv2a = Conv2D(256,(3,3),strides=(2,2),padding='valid',name='conv2a')(padding)
    conv2a = LeakyReLU(0.1)(conv2a)
    conv2a_1 = Conv2D(256,(3,3),padding='same',name='conv2a_1')(conv2a)
    conv2a_1 = LeakyReLU(0.1)(conv2a_1)
    padding = ZeroPadding2D()(conv2a_1)

    conv3 = Conv2D(256,(3,3),strides=(2,2),padding='valid',name='conv3')(padding)
    conv3 = LeakyReLU(0.1)(conv3)
    conv3_1 = Conv2D(256,(3,3),padding='same',name='conv3_1')(conv3)
    conv3_1 = LeakyReLU(0.1)(conv3_1)
    padding = ZeroPadding2D()(conv3_1)

    conv4 = Conv2D(512,(3,3),strides=(2,2),padding='valid',name='conv4')(padding)
    conv4 = LeakyReLU(0.1)(conv4)
    conv4_1 = Conv2D(512,(3,3),padding='same',name='conv4_1')(conv4)
    conv4_1 = LeakyReLU(0.1)(conv4_1)
    padding = ZeroPadding2D()(conv4_1)

    conv5 = Conv2D(512,(3,3),strides=(2,2),padding='valid',name='conv5')(padding)
    conv5 = LeakyReLU(0.1)(conv5)
    conv5_1 = Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv5_1')(conv5)
    conv5_1 = LeakyReLU(0.1)(conv5_1)
    padding = ZeroPadding2D()(conv5_1)

    conv7 =  Conv2D(512,(3,3),padding='same')(padding)
    conv7 = LeakyReLU(0.1)(conv7)
    padding = ZeroPadding2D()(conv7)

    conv7a =  Conv2D(512,(3,3),padding='same')(padding)
    conv7a = LeakyReLU(0.1)(conv7a)
    padding = ZeroPadding2D()(conv7a)

    conv8 = Conv2D(256,(3,3),padding='same')(conv7a)
    conv8 = LeakyReLU(0.1)(conv8)
    padding = ZeroPadding2D()(conv8)

    conv9 = Conv2D(64,(3,3),padding='same')(conv8)
    conv9 = LeakyReLU(0.1)(conv9)
    dense = Flatten()(conv9)
    dense = Dense(64, activation="relu")(dense)
    dense = Dense(16, activation="relu")(dense)
    dense = Dense(3, activation="tanh")(dense)
    model = Model(x,dense)
    #model.summary()
    return model

def resize(scale_percent, src):

    #calculate the 30 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    # resize image
    output = cv2.resize(src, dsize)
    return output

def func (i):
    global frames
    im1 = frames[i]
    im2 = frames[i+1]
    im1 = resize(30, im1)
    im2 = resize(30, im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.


    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    return flow

dfile = pd.read_csv("/workspace/storage/T_mini.csv")
dfile.head()
name = np.array(dfile['File_Name'])
X = np.array(dfile['Camera_x'])
Y = np.array(dfile['Camera_y'])
Z = np.array(dfile['Camera_z'])
print(X[0], Y[0], Z[0])

#define generators

class DataGenerator(Sequence):

  def __init__(self, batch_size=16, to_fit=True, shuffle=True):
    self.batch_size = batch_size
    self.to_fit = to_fit
    self.shuffle = shuffle
    self.on_epoch_end()

  def on_epoch_end(self):
    self.indexes = np.arange(len(name)-100000)
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __len__(self):
    return 500
    #return int(np.floor((len(name)-100000)/self.batch_size))
  
  def __getitem__(self, iter):
    data = []
    gt = []
    #iter = iter%(len(name)-100000)
    indexes = self.indexes[iter*self.batch_size:(iter+1)*self.batch_size]
    for i in indexes:
      if not (os.path.isfile("/workspace/storage/Tf/"+name[i]+".npy")):
        global frames
        frames = []
        vid = cv2.VideoCapture("/workspace/storage/T/"+name[i]+".avi")
        tmp = np.array([X[i], Y[i], Z[i]])
        gt.append(tmp)
        while(True):
            ret, frame = vid.read()
            if ret:
              frames.append(frame)
            else:
              break
        #Optical flow conversion
        global dataset
        agents = 30
        chunksize = 1
        with Pool(processes=agents) as pool:
            result = pool.map(func, dataset, chunksize)
        video = np.asarray(result)
        video = np.transpose(video, (1,2,0,3))  
        video = np.reshape(video, (144, 192, 58))
        path="/workspace/storage/Tf/"+name[i]+".npy"
        np.save(path,video)
      flow= np.load("/workspace/storage/Tf/"+name[i]+".npy")
      data.append(flow_video)
    x = np.asarray(data)
    y = np.asarray(gt)
    if(self.to_fit):
      return (x,y)
    else:
      return x

class ValDataGenerator(Sequence):

  def __init__(self, batch_size=16, to_fit=True, shuffle=True):
    self.batch_size = batch_size
    self.to_fit = to_fit
    self.shuffle = shuffle
    self.on_epoch_end()

  def on_epoch_end(self):
    self.indexes = np.arange(len(name)-100000, len(name))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __len__(self):
    return 100
  
  def __getitem__(self, iter):
    data = []
    gt = []
    iter = (iter%100000)
    indexes = self.indexes[iter*self.batch_size:(iter+1)*self.batch_size]
    for i in indexes:
      if not (os.path.isfile("/workspace/storage/Tf/"+name[i]+".npy")):
        global frames
        frames = []      
        vid = cv2.VideoCapture("/workspace/storage/T/"+name[i]+".avi")
        tmp = np.array([X[i], Y[i], Z[i]])
        gt.append(tmp)
        while(True):
          ret, frame = vid.read()
          if ret:
            frames.append(frame)
          else:
            break
        #Optical flow conversion
        global dataset
        agents = 30
        chunksize = 1
        with Pool(processes=agents) as pool:
            result = pool.map(func, dataset, chunksize)
        video = np.asarray(result)
        video = np.transpose(video, (1,2,0,3))
        video = np.reshape(video, (144, 192, 58))
        path="/workspace/storage/Tf/"+name[i]+".npy"
        np.save(path,video)
      flow= np.load("/workspace/storage/Tf/"+name[i]+".npy")
      data.append(flow_video)
    x = np.asarray(data)
    y = np.asarray(gt)
    if(self.to_fit):
      return (x,y)
    else:
      return x

training_generator = DataGenerator()
validation_generator = ValDataGenerator()

#initialise model
model = k_model([144, 192, 58])
opt = keras.optimizers.Adam(learning_rate=0.000001)
model.compile(loss="mse", optimizer=opt, metrics=["mse"])

logdir = os.path.join("/workspace/storage/tensorboard_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
filepath = "/workspace/storage/checkpoints/model_flow.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, tensorboard_callback]

model.fit(training_generator, validation_data=validation_generator, verbose=1, epochs=1000,
          shuffle=True, callbacks = callbacks_list)
