from reconstructor import Reconstructor
import utils
import numpy as np
import cv2
import hull


img = np.zeros([1280, 800, 3], np.uint8)


Recon = Reconstructor(utils.loadCameraParameters(), 10, utils.getImageStack('rubiks'))
cube = utils.createCubeCorners([-30, -30, 30], 60)

projcubes = Recon.projectPointsToAllViews(cube)
for pcube in projcubes:
    pts = hull.createHullPoints(pcube)
    img, binimg = hull.drawHull(pts, (100, 1000))
    cv2.imshow('img', img)
    key = cv2.waitKey(200)
    '''
    img2 = img.copy()
    pts = np.array([cv2.convexHull(pcube)[:, 0]])
    binimg = cv2.drawContours(img2, pts, -1, thickness=cv2.FILLED, color=(255, 255, 255))
    binimg = np.array(binimg, np.bool)
    cv2.imshow('img', 255 * np.array(binimg, np.uint8))
    key = cv2.waitKey(200)
    '''
while True:
    key = cv2.waitKey(20)
    if key == 27:
        break