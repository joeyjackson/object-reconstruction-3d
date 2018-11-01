import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi
import math
import heapq


def rotation_matrix(axis, theta):
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


def get_rays(img_dir, max_count, window_name):
    cameras = []
    rays = []
    count = 1
    focal_length = 1185.5
    for f in os.listdir(img_dir):
        if f[f.rfind('.'):] == '.jpg':
            theta = int(f[:f.rfind('.')]) * pi / 180
            camera_pos = np.array(cylindrical2cartesian(31.5, theta, 25.5), dtype=np.float32)
            camera_direction = 5 * np.array(normalize(cylindrical2cartesian(25.5 / np.tan(38.99 * pi / 180),
                                                                            theta + pi, -25.5)), dtype=np.float32)
            cameras.append((camera_pos, camera_direction))
            filename = os.path.join('testImg', f)
            img = cv2.imread(filename)
            corner_list = get_corners(img, window_name)  # N x 2
            corner_list = corner_list.T  # 2 x N
            corner_list = np.array([corner_list[1] - int(img.shape[1] / 2), corner_list[0] - int(img.shape[0] / 2),
                                    np.ones(len(corner_list[0])) * focal_length])  # 3 x N
            corner_list = corner_list.T  # N x 3
            pointsTransformed = np.array([[pt] for pt in corner_list])
            # intrinsic = np.array([[1920, 0, int(img.shape[1] / 2)],
            #                       [0, 1080, int(img.shape[0] / 2)],
            #                       [0, 0, 1]])
            rot1 = rotation_matrix([1, 0, 0], (128) * pi / 180)
            pointsTransformed = cv2.transform(pointsTransformed, rot1)
            rot2 = rotation_matrix([0, 0, 1], (-pi / 2) + theta)
            pointsTransformed = cv2.transform(pointsTransformed, rot2)
            for ray in pointsTransformed:
                ray1 = ray[0]
                rays.append((camera_pos, 30 * np.array(normalize(ray1))))
        if count >= max_count:
            break
        else:
            count += 1
    return rays, cameras


def main():
    WIN_NAME = 'CORNERS'

    rays, cameras = get_rays('testImg', 36, WIN_NAME)
    origin = (40, 40, 1)
    space_size = (80, 80, 31)
    votes = vector_voting(rays, space_size=space_size, origin=origin)

    max_vote = np.max(votes)
    votes = votes.astype(np.double) / max_vote
    pts = []
    for r in range(votes.shape[0]):
        for c in range(votes.shape[1]):
            for p in range(votes.shape[2]):
                # # TOP K
                # k = 30
                # if len(pts) < k:
                #     heapq.heappush(pts, [votes[r, c, p],
                #                          int(c) - origin[0], int(r) - origin[1], int(p) - origin[2]])
                # else:
                #     heapq.heappushpop(pts, [votes[r, c, p],
                #                             int(c) - origin[0], int(r) - origin[1], int(p) - origin[2]])
                if votes[r, c, p] > 0:
                    pts.append([int(c) - origin[0], int(r) - origin[1], int(p) - origin[2]])
    # pts = [[pt[1], pt[2], pt[3]] for pt in pts]
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    zs = [pt[2] for pt in pts]

    fig = plt.figure()
    ax_scat = fig.add_subplot(111, projection='3d')
    ax_scat.scatter(xs, ys, zs, c='r', marker='o')
    ax_scat.set_xlim([-40., 40.])
    ax_scat.set_ylim([-40., 40.])
    ax_scat.set_zlim([-1., 30.])
    plt.show()

    # ax_vect = fig.add_subplot(122, projection='3d')
    #
    # rays_flat = [(x, y, z, u, v, w) for ((x, y, z), (u, v, w)) in rays]
    # rX, rY, rZ, rU, rV, rW = zip(*rays_flat)
    # cameras_flat = [(x, y, z, u, v, w) for ((x, y, z), (u, v, w)) in cameras]
    # cX, cY, cZ, cU, cV, cW = zip(*cameras_flat)
    #
    # ax_vect.quiver(rX, rY, rZ, rU, rV, rW)
    # # ax.quiver(cX, cY, cZ, cU, cV, cW)
    # ax_vect.set_xlim([-40., 40.])
    # ax_vect.set_ylim([-40., 40.])
    # ax_vect.set_zlim([-1., 30.])
    # plt.show()
    cv2.destroyAllWindows()


def ray_vote(space, ray, origin):
    def in_space(pt):
        pt = pt + origin
        x, y, z = pt
        rows, cols, planes = space.shape
        return 0 <= x < cols and 0 <= y < cols and 0 <= z < planes

    start, dir = ray
    dir = normalize(dir)

    curr = start
    for i in range(100):
        if in_space(curr):
            space[int(round(curr[1] + origin[1])),
                  int(round(curr[0] + origin[0])),
                  int(round(curr[2] + origin[2]))] += 1
        curr += dir

    # pts = []
    # for r in range(space.shape[0]):
    #     for c in range(space.shape[1]):
    #         for p in range(space.shape[2]):
    #             if space[r, c, p] > 0:
    #                 pts.append([int(c)-origin[0], int(r)-origin[1], int(p)-origin[2]])
    # xs = [pt[0] for pt in pts]
    # ys = [pt[1] for pt in pts]
    # zs = [pt[2] for pt in pts]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # ax.set_xlim([-40., 40.])
    # ax.set_ylim([-40., 40.])
    # ax.set_zlim([-1., 30.])
    # ax.scatter(xs, ys, zs, c='r', marker='o')
    # plt.show()


def vector_voting(rays, space_size, origin):
    voting_space = np.zeros(space_size)
    for ray in rays:
        ray_vote(voting_space, ray, origin)
    return voting_space


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
