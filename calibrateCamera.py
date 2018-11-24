import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cal_imgs = glob.glob("cal/*.jpg")  # get all calibration image files
cal_dim = (6, 9)  # dimensions of desired calibration grid in points, on an arbitrary axis
scale = 20  # scale of the calibration coordinate system in mm
d3_points = []  # the true 3D coordinates in each calibration image. Holds copies of calp
d2_points = []  # the resulting 2D pixel coordinates of each calibration point.

ax1 = np.tile(np.arange(0, cal_dim[0])[:, np.newaxis], (cal_dim[1],1))  # create 'x' coordinates
ax2 = np.repeat(np.arange(0, cal_dim[1]), cal_dim[0])[:, np.newaxis]  # create 'y' coordinates
ax3 = np.zeros((cal_dim[0]*cal_dim[1],1)) # create 'z' coordinates
calp = scale * np.float32(np.concatenate([ax1, ax2, ax3], 1))  # get the points the calibration image holds

for imgname in cal_imgs:
    img = cv2.cvtColor(cv2.imread(imgname), cv2.COLOR_BGR2GRAY)  # load a single cal image in grey scale
    r, corners = cv2.findChessboardCorners(img, cal_dim, None)  # find calibration corners

    if r == True:  # we found corners successfully
        cornerpts = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

        d2_points.append(cornerpts)
        d3_points.append(calp)

_, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(d3_points, d2_points, img.shape[::-1], None, None)

np.savez('Camera_Calibration', cam_matrix=mtx, distortion=dist)  # save camera properties
