import  pickle
import cv2

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
    # break
    # prvs = next