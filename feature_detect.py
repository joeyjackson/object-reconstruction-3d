import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi


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
    for f in os.listdir('testImg'):
        if f[f.rfind('.'):] == '.jpg':
            theta = int(f[:f.rfind('.')]) * pi/180
            camera_pos = np.array(cylindrical2cartesian(31.5, theta, 25.5), dtype=np.float32)
            camera_direction = np.array(normalize(cylindrical2cartesian(25.5/np.tan(38.99 * pi/180),
                                                                        theta + pi, -25.5)), dtype=np.float32)
            cameras.append((camera_pos, camera_direction))
            filename = os.path.join('testImg', f)
            img = cv2.imread(filename)
            corner_list = get_corners(img, WIN_NAME)
            corner_list = np.array([corner_list[0], corner_list[1], np.ones(len(corner_list[0]))]).T
            corner_list = np.array([[pt] for pt in corner_list])
            intrinsic = np.array([[1600, 0, int(img.shape[1] / 2)],
                                  [0, 1600, int(img.shape[0] / 2)],
                                  [0, 0, 1]])
            pointsTransformed = cv2.transform(corner_list, np.invert(intrinsic))
            # print(pointsTransformed)
            for ray in pointsTransformed:
                ray1 = ray[0]
                rays.append( (camera_pos, 100 * np.array(normalize(ray1))) )


    cameras_flat = [(x, y, z, u, v, w) for ((x, y, z), (u, v, w)) in cameras]
    X, Y, Z, U, V, W = zip(*cameras_flat)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W)
    ax.set_xlim([-40., 40.])
    ax.set_ylim([-40., 40.])
    ax.set_zlim([-1., 30.])
    plt.show()
    cv2.destroyAllWindows()


def get_corners(img, window_name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = np.where(dst > 0.8 * dst.max())
    img[corners] = [255, 0, 255]
    cv2.imshow(window_name, cv2.resize(img, (int(img.shape[0] / 2), int(img.shape[0] / 2))))
    # if cv2.waitKey(0) & 0xff == 27:
    #     pass
    return corners


if __name__ == '__main__':
    main()
