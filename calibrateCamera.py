import numpy as np
import cv2
import glob

# USER PARAMETERS
calibration_directory = 'cal'  # folder where all the calibration images are stored
scale = 12  # scale of the calibration coordinate system in mm
dtheta = 10  # rotation of every image used for calibration
cal_dim = (9, 6)  # dimensions of desired calibration grid in points, on an arbitrary axis
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # calibration criteria

# Parameters setup
xoffset = scale*(cal_dim[0] - 1)/2  # offset of the corner point to the rotation center, x axis
yoffset = scale*(cal_dim[1] - 1)/2  # offset of the corner point to the rotation center, y axis

cal_imgs = glob.glob(calibration_directory + '\*.jpg')  # get all calibration image files
cal_imgs.sort()  # sort into numerical order

d3_points = []  # the true 3D coordinates in each calibration image. Holds copies of calp
d2_points = []  # the resulting 2D pixel coordinates of each calibration point.

# 3D Coordinate grid creation
ax1 = scale * np.tile(np.arange(0, cal_dim[0])[:, np.newaxis], (cal_dim[1],1)) - xoffset # create 'x' coordinates
ax2 = scale * np.repeat(np.arange(0, cal_dim[1]), cal_dim[0])[:, np.newaxis] - yoffset  # create 'y' coordinates
ax3 = np.zeros((cal_dim[0]*cal_dim[1],1))  # create 'z' coordinates

calp = np.float32(np.concatenate([ax1, ax2, ax3], 1))  # get the points the calibration image holds

# Calibration loop
angles = []
for imgname in cal_imgs:
    imcol = cv2.imread(imgname)
    img = cv2.cvtColor(imcol, cv2.COLOR_BGR2GRAY)  # load a single cal image in grey scale
    r, corners = cv2.findChessboardCorners(img, cal_dim, None)  # find calibration corners

    if r == True:  # we found corners successfully
        angles.append(10*(int(imgname[-6:-4]) - 1))  # record the angle the shot was taken at for housekeeping
        cornerpts = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)  # refine calibration corners

        d2_points.append(cornerpts)  # where do the chessboard points lie in the 2D image?
        d3_points.append(calp)  # where do the chessboard points lie in reality, based on our arbitrary 3D coordinate?

        cv2.drawChessboardCorners(imcol, (9, 6), cornerpts, r)  # display corners and show them to the user
        cv2.imshow('img', imcol)
        cv2.waitKey(200)

_, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(d3_points, d2_points, img.shape[::-1], None, None)

np.savez('Camera_Calibration', cam_matrix=mtx, distortion=dist, rvecs=rvecs, tvecs=tvecs, angles=angles)  # save camera properties
