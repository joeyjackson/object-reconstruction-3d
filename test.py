from reconstructor import Reconstructor
import utils
import numpy as np
import cv2
import hull
import pickle
import imageio

imgs = utils.getImageStack('temp1')
imageio.mimsave('rubiks2.gif', imgs, duration=0.06)
#Recon = pickle.load(open('obj/rubiks2.obj', 'rb'))
#Recon.rotateModel('temp1' + '/' + 'rubiks2')
'''
imstack = utils.getImageStack('captures/hexb')
imageio.mimsave('hexb' + '.gif', imstack, duration=0.06)

imstack = utils.getImageStack('captures/hex')
imageio.mimsave('hex' + '.gif', imstack, duration=0.06)

imstack = utils.getImageStack('captures/clothespinb')
imageio.mimsave('pinb' + '.gif', imstack, duration=0.06)

imstack = utils.getImageStack('captures/pin')
imageio.mimsave('pin' + '.gif', imstack, duration=0.06)

imstack = utils.getImageStack('captures/rubiks')
imageio.mimsave('rubiks' + '.gif', imstack, duration=0.06)

imstack = utils.getImageStack('captures/rubiksb')
imageio.mimsave('rubiksb' + '.gif', imstack, duration=0.06)

imstack = utils.getImageStack('captures/rubiks2')
imageio.mimsave('rubiks2' + '.gif', imstack, duration=0.06)


imstack = utils.getImageStack('captures/rubiks2b')
imageio.mimsave('rubiks2b' + '.gif', imstack, duration=0.06)
'''

#imstack = utils.getImageStack('captures/spray')
#imageio.mimsave('spray' + '.gif', imstack, duration=0.06)
'''
Recon = pickle.load(open('obj/spray6.obj', 'rb'))
binimgstack = Recon.masks
imstack = []
for binimg in binimgstack:
    sz = binimg.shape
    img = np.zeros((sz[0],sz[1],3), np.uint8)
    img[binimg,:] = 255
    imstack.append(img)

'''
#imageio.mimsave('sprayb' + '.gif', imstack, duration=0.06)
'''
Recon = pickle.load(open('obj/cube1.obj', 'rb'))
Recon.rotateModel('temp/test')

imgs = utils.getImageStack('temp')

video = cv2.VideoWriter('cube1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

for img in imgs:
    video.write(img)

video.release()
'''
'''
Recon = pickle.load(open('obj/clip.obj', 'rb'))
cube = utils.createCubeCorners(((-40, -40, -20), 10))

#video = cv2.VideoWriter('projcube2.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, (800,1280))

stack = []
img = np.zeros([1280, 800, 3], np.uint8)
for x in range(-40, 10, 10):
        for z in range(-80, 0, 10):
            cube = utils.createCubeCorners(((x, x, z), 10))
            projcubes = Recon.projectPointsToAllViews(cube)


            for pcube in projcubes:
                pts = hull.createHullPoints(pcube)
                img, binimg = hull.drawHull(pts, (1280,800))

                stack.append(img)
                #video.write(img.copy())
                #key = cv2.waitKey(1)


imageio.mimsave('projcube3' + '.gif', stack, duration=0.0012)
#video.release()
#while True:
#    key = cv2.waitKey(20)
#    if key == 27:
#        break
'''