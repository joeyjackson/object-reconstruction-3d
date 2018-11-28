from reconstructor import Reconstructor
import utils
import numpy as np
import cv2
import hull
import pickle


Recon = pickle.load(open('obj/cube1.obj', 'rb'))
Recon.rotateModel('temp/test')

imgs = utils.getImageStack('temp')

video = cv2.VideoWriter('cube1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

for img in imgs:
    video.write(img)

video.release()
'''
Recon = pickle.load(open('reconTest.obj', 'rb'))
cube = utils.createCubeCorners(((-40, -40, -20), 10))

video = cv2.VideoWriter('projcube2.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, (800,1280))


img = np.zeros([1280, 800, 3], np.uint8)
for x in range(-40,0,20):
        for z in range(-80,0,20):
            cube = utils.createCubeCorners(((x, 0, z), 20))
            projcubes = Recon.projectPointsToAllViews(cube)


            for pcube in projcubes:
                pts = hull.createHullPoints(pcube)
                img, binimg = hull.drawHull(pts, (1280,800))

                video.write(img.copy())
                #key = cv2.waitKey(1)

video.release()
#while True:
#    key = cv2.waitKey(20)
#    if key == 27:
#        break
'''