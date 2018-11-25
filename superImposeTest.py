import numpy as np
import cv2
import utils

cam_mtx, cam_dist, tvec, rvec = utils.loadCameraParameters()

vc = cv2.VideoCapture(0)
vc.set(3, 1920)
vc.set(4, 1080)

'''
cube = np.array([[30, 30, 0],
                 [0, 30, 0],
                 [30, 0, 0],
                 [0, 0, 0],
                 [30, 30, -30],
                 [0, 30, -30],
                 [30, 0, -30],
                 [0, 0, -30]])
'''
cube = np.array([[30, 30, 0],
                 [0,0,0],
                [-30, 30, 0],
                [-30, -30, 0],
                [30, -30, 0],
                [30, 30, -60],
                [-30, 30, -60],
                [-30, -30, -60],
                [30, -30, -60]], np.float32)



img = cv2.imread('cal/Calibrator01.jpg')

if vc.isOpened():
    ret, img = vc.read()
else:
    ret = False

#THE ANGLE COORDINATE SYSTEM IS DEFINED IN A COUNTER-CLOCKWISE FASHION
#video = cv2.VideoWriter('SampleCube1.avi', cv2.VideoWriter_fourcc(*'XVID'), 60.0, (800, 1280))
for angle in range(0, 360, 1):
    while ret:
        ret, img = vc.read()
        if ret:
            img = utils.flipImage(img)
            break
    img2 = img.copy()
    corners = utils.rotatePointsAboutZ(angle, cube)
    newpt, _ = cv2.projectPoints(corners, rvec, tvec, cam_mtx, cam_dist)

    for pts in newpt:
        cv2.circle(img2, tuple(pts[0]), 3, (0, 0, 255), 2)

    cv2.imshow('img', img2)
    #video.write(img2)
    key = cv2.waitKey(1)

#video.release()
