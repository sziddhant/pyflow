from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2
import pickle
import multiprocessing
from multiprocessing import Pool


num_cores = multiprocessing.cpu_count()

cap = cv2.VideoCapture("Blender.avi")
ret, im1 = cap.read()

frames = []
frames.append(im1)
while(1):
    ret, im1 = cap.read()
    if ret:
        #cv2.imshow('frame2', im1)
        #cv2.waitKey(0)
        frames.append(im1)
    else:
        break

#for i in range(len(frames)-1):
def func (i):
    im1 = frames[i]
    im2 = frames[i+1]
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

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    #e = time.time()
    #print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
    #    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    # np.save('examples/outFlow.npy', flow)

    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb
    print(i)

dataset=[i for i in range(0,29)]

agents = 4
chunksize = 1
with Pool(processes=agents) as pool:
  result = pool.map(func, dataset, chunksize)



#print (result.shape)
with open("Blender.txt", "wb") as fp:
    #result.tolist()
    pickle.dump(result, fp)

with open("Blender.txt", "rb") as fp:
    res = pickle.load(fp)

for i in range(len(res)):
    rgb = res[i]
    cv2.imshow('frame2', rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)

