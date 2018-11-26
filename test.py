from reconstructor import Reconstructor
import utils
import numpy as np
import cv2
import hull
import pickle

img = np.zeros([1280, 800, 3], np.uint8)

Recon = pickle.load(open('reconTest.obj', 'rb'))
#Recon = Reconstructor(utils.loadCameraParameters(), 10, utils.getImageStack('rubiks'))
#Recon.save('reconTest')
subcube = utils.cubeOctsect(((-30, -30, 0), 60))
cube = utils.createCubeCorners(subcube[0])
cube2 = utils.createCubeCorners(((10, 10, 50), 30))


projcubes = Recon.projectPointsToAllViews(cube)
projcube2 = Recon.projectPoints(cube2, 10)
for pcube in projcubes:
    pts = hull.createHullPoints(pcube)
    img, binimg = hull.drawHull(pts, (1000, 1000))

    pts2 = hull.createHullPoints(projcube2)
    _, binimg2 = hull.drawHull(pts2, (1000, 1000))

    print(Recon.intersectStatus(binimg, binimg2))
    cv2.imshow('img', img)
    key = cv2.waitKey(20)
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