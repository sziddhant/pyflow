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

dfile = pd.read_csv("/workspace/storage/T_mini.csv")
dfile.head()
name = np.array(dfile['File_Name'])



num_cores = multiprocessing.cpu_count()
print(num_cores)
for ii in range(300000):
    s = time.time()

    cap = cv2.VideoCapture("/workspace/storage/T/"+name[i]+".avi")
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

        
        u, v, im2W = pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        
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

    agents = 29
    chunksize = 1
    with Pool(processes=agents) as pool:
        result = pool.map(func, dataset, chunksize)

    out = cv2.VideoWriter("/workspace/storage/Tf/"+name[i]+".avi",  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, (640,480))   
    for i in range(len(result)):
        rgb = result[i]
        out.write(rgb)
    out.release()
    cap.release()
    e = time.time()
    print("%d in %.2fs"% (ii,e-s))
    