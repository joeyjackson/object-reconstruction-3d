import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi
import math


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def cartesian2cylindrical(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi, z)


def cylindrical2cartesian(rho, phi, z):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y, z)


def normalize(vector):
    s = np.sqrt(sum([x**2 for x in vector]))
    return [x/s for x in vector]


def main():
    WIN_NAME = 'CORNERS'
    cameras = []
    rays = []
    count = 1
    focal_length = 1185.5
    for f in os.listdir('testImg'):
        if f[f.rfind('.'):] == '.jpg':
            theta = int(f[:f.rfind('.')]) * pi/180
            camera_pos = np.array(cylindrical2cartesian(31.5, theta, 25.5), dtype=np.float32)
            camera_direction = 10 * np.array(normalize(cylindrical2cartesian(25.5/np.tan(38.99 * pi/180),
                                                                        theta + pi, -25.5)), dtype=np.float32)
            cameras.append((camera_pos, camera_direction))
            filename = os.path.join('testImg', f)
            img = cv2.imread(filename)
            corner_list = get_corners(img, WIN_NAME)  # N x 2
            corner_list = corner_list.T  # 2 x N
            corner_list = np.array([corner_list[1] - int(img.shape[1] / 2), corner_list[0] - int(img.shape[0] / 2),
                                    np.ones(len(corner_list[0])) * focal_length])  # 3 x N
            corner_list = corner_list.T  # N x 3
            pointsTransformed = np.array([[pt] for pt in corner_list])
            # intrinsic = np.array([[1920, 0, int(img.shape[1] / 2)],
            #                       [0, 1080, int(img.shape[0] / 2)],
            #                       [0, 0, 1]])
            rot1 = rotation_matrix([1, 0, 0], (128) * pi/180)
            pointsTransformed = cv2.transform(pointsTransformed, rot1)
            rot2 = rotation_matrix([0, 0, 1], (-pi/2)+theta)
            pointsTransformed = cv2.transform(pointsTransformed, rot2)
            for ray in pointsTransformed:
                ray1 = ray[0]
                rays.append( (camera_pos, 30 * np.array(normalize(ray1))) )
        if count >= 36:
            break
        else:
            count += 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    rays_flat = [(x, y, z, u, v, w) for ((x, y, z), (u, v, w)) in rays]
    rX, rY, rZ, rU, rV, rW = zip(*rays_flat)
    cameras_flat = [(x, y, z, u, v, w) for ((x, y, z), (u, v, w)) in cameras]
    cX, cY, cZ, cU, cV, cW = zip(*cameras_flat)


    ax.quiver(rX, rY, rZ, rU, rV, rW)
    # ax.quiver(cX, cY, cZ, cU, cV, cW)
    ax.set_xlim([-40., 40.])
    ax.set_ylim([-40., 40.])
    ax.set_zlim([-1., 30.])
    plt.show()
    cv2.destroyAllWindows()


def filter_blocks(mask, kernel, threshold=0.1):
    centers = []
    high = mask.max()
    height, width = mask.shape
    for r in range(0, height, kernel[0]):
        for c in range(0, width, kernel[1]):
            center = [r + int(kernel[0] / 2), c + int(kernel[1] / 2)]
            window = mask[r: r + kernel[0]+1, c: c + kernel[1]+1]
            if window.max() > threshold * high:
                centers.append(center)
    return np.array(centers)


def get_corners(img, window_name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # corners = np.where(dst > 0.8 * dst.max())
    corners = filter_blocks(dst, (7, 7))
    for corner in corners:
        # img[corner[0], corner[1], :] = [255, 0, 255]
        cv2.circle(img, (corner[1], corner[0]), 8, (255, 0, 255), 2)
    cv2.imshow(window_name, cv2.resize(img, (int(img.shape[0] / 2), int(img.shape[0] / 2))))
    # if cv2.waitKey(0) & 0xff == 27:
    #     pass
    return corners


if __name__ == '__main__':
    main()
