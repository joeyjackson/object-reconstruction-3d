import numpy as np
import glob
import cv2


def flipImage(img):
    # Flips an image such that it turns from landscape to portrait or vice versa
    #
    # ---- Inputs ----------------------------
    # img = An image (3D array) either from camera stream or jpg load.
    #
    # ----- Outputs ---------------------------
    # returns an image (3D array) with its x and y axes flipped. Turns a landscape image portrait
    #
    #
    r1 = np.transpose(img[:, :, 0])[:, :, np.newaxis]  # flip an image to portrait
    r2 = np.transpose(img[:, :, 1])[:, :, np.newaxis]
    r3 = np.transpose(img[:, :, 2])[:, :, np.newaxis]

    return np.concatenate([r1, r2, r3], 2)


def rotatePointsAboutZ(angle, points):
    # Rotates multiple points about the Z axis
    #
    # ---- Inputs ----------------------------
    # angle = the angle to rotate, in degrees
    # points = a list of lists of length 3, tuples of length 3, or a 2D numpy array where each row is length 3
    #          Each coordinate is described as x, y, z
    #          [[x,y,z],[x,y,z]...]
    #
    # ----- Outputs ---------------------------
    # returns a 2D float32 numpy array where each row is the corresponding input coordinate rotated about Z by angle.
    #
    #
    newpts =[]
    for pt in points:
        newpts.append(rotatePointAboutZ(angle, pt))

    newpts = np.array(newpts, np.float32)
    return newpts


def rotatePointAboutZ(angle, point):
    # Rotates a single point about the Z axis
    #
    # ---- Inputs ----------------------------
    # angle = the angle to rotate, in degrees
    # point = a coordinate which is a list of length 3, tuple of length 3, or a length 3 1D lumpy array
    #          Each coordinate is described as x, y, z
    #          [x,y,z]
    #
    # ----- Outputs ---------------------------
    # returns a length 3 1D numpy array where each row is the corresponding input coordinate rotated about Z by angle.
    #
    #
    rt = angle*np.pi/180  # angle to radians

    rmtx = np.array([[np.cos(rt), -np.sin(rt), 0],
                     [np.sin(rt), np.cos(rt), 0],
                     [0, 0, 1]])  # Construct rotation matrix about Z

    p = np.array(point, np.float32)  # if the input point was not already an numpy float32 array
    p = p[:, np.newaxis]  # force the 1x3 coordinate it into a 3x1 vector so it can be multiplied by the rotation mtx

    newp = np.asmatrix(rmtx)*np.asmatrix(p)  # get rotated coordinate

    newp = np.array(newp, np.float32)  # make sure the output is a numpy float32 array

    return np.squeeze(newp[:, 0])  # return a 1x3 rotated coordinate


def loadCameraParameters(fname='Camera_Calibration.npz'):
    # Get all camera properties needed for backprojection: essential matrix, distortion, position, rotation
    #
    # ---- Inputs ----------------------------
    # fname = string name of the npz file that holds the camera calibration properties
    #
    # ----- Outputs ---------------------------
    # returns (cam_matrix, cam_distortion, pos, rot)
    # cam_matrix = 3x3 camera essential matrix
    # cam_distortion = 5-variable distortion of the camera
    # pos = 1x3 Position of the camera as determined by the calibration
    # rot = 1x3 Rotation of the camera as determined by the calibration
    #
    #
    params = np.load(fname)
    cam_matrix = params['cam_matrix']
    cam_distortion = params['distortion']
    rvecs = params['rvecs']
    tvecs = params['tvecs']

    pos = np.mean(tvecs,0)
    rot = rvecs[0]

    return cam_matrix, cam_distortion, pos, rot


def getImageStack(directory):
    # Reads some numbered images and sorts them based on their numbering, then places them in order into a list
    #
    # ---- Inputs ----------------------------
    # directory = string of the folder name to look into. Do NOT use a '\'
    #
    # ----- Outputs ---------------------------
    # imageStack = list of numpy images that contains all the jpg images contained within the folder
    #
    #
    imnames = glob.glob(directory + '/*')  # get all calibration image files
    imnames.sort()  # sort into numerical order

    imageStack = []
    for imname in imnames:
        imageStack.append(cv2.imread(imname))

    return imageStack


def createCubeCorners(cube):
    # given the minima corner of a cube and a side length in mm, returns a 2D numpy array of the 8 corners of the cube
    #
    # ---- Inputs ----------------------------
    # v = 3D coordinate in a list, tuple, or numpy form [x,y,z]
    #     of a cube corner that has the minimum x, y, and -z coordinate
    # s = cube side length in mm
    #
    # ----- Outputs ---------------------------
    # corners = 8x3 numpy array containing the 8 corners of the cube.
    #
    #
    v = cube[0]
    s = cube[1]

    corners = []
    corners.append([v[0], v[1], v[2]])
    corners.append([v[0] + s, v[1], v[2]])
    corners.append([v[0] + s, v[1] + s, v[2]])
    corners.append([v[0], v[1] + s, v[2]])
    corners.append([v[0], v[1], v[2] - s])
    corners.append([v[0] + s, v[1], v[2] - s])
    corners.append([v[0] + s, v[1] + s, v[2] - s])
    corners.append([v[0], v[1] + s, v[2] - s])

    corners = np.array(corners)
    return corners


def cubeOctsect(cube):
    c = cube[0]
    sz = cube[1]

    newsz = sz/2

    newcubes = []
    newcubes.append((c, newsz))
    newcubes.append(((c[0]+newsz, c[1], c[2]), newsz))
    newcubes.append(((c[0], c[1]+newsz, c[2]), newsz))
    newcubes.append(((c[0] + newsz, c[1]+newsz, c[2]), newsz))
    newcubes.append(((c[0], c[1], c[2]-newsz), newsz))
    newcubes.append(((c[0] + newsz, c[1], c[2]-newsz), newsz))
    newcubes.append(((c[0], c[1] + newsz, c[2]-newsz), newsz))
    newcubes.append(((c[0] + newsz, c[1] + newsz, c[2]-newsz), newsz))

    return newcubes


