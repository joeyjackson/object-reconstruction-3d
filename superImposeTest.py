import numpy as np
import cv2
from loadCamera import loadCameraParameters

cam_mtx, cam_dist = loadCameraParameters('Camera_Calibration.npz')
print(cam_mtx)
print(cam_dist)

#in progress