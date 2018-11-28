import cv2
import utils
import segment
import hull
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


class Reconstructor:

    def __init__(self, params, angle, imgstack, init_cube):
        self.cam_mtx = params[0]  # camera matrix
        self.cam_dist = params[1]  # camera distortion
        self.cam_pos = params[2]  # camera position
        self.cam_rot = params[3]  # camera rotation

        self.increment = angle  # rotation per captured image in degrees

        self.masks = []  # binary foreground masks
        self.imsize = imgstack[0].shape
        for img in imgstack:
            self.masks.append(segment.segment(img))  # get binary silhouette of foreground

        self.init_cube = init_cube
        self.model = []

    def projectPoints(self, points, angle):

        rotated_points = utils.rotatePointsAboutZ(angle, points)
        impts, _ = cv2.projectPoints(rotated_points, self.cam_rot, self.cam_pos, self.cam_mtx, self.cam_dist)
        impts = impts[:, 0]  # cv2 keeps outputting 3D numpy arrays
        impts = np.array(np.round(impts), int)  # round to pixel indices
        return impts

    def projectPointsToAllViews(self, points):

        viewpts = []
        for idx in range(0, len(self.masks)):
            angle = self.increment * idx
            viewpts.append(self.projectPoints(points, angle))

        return viewpts

    def intersectStatus(self, bhull, bview, tolerance=0.025):
        hullct = np.count_nonzero(bhull)  # count hull area

        intersect = np.logical_and(bhull, bview)  # overlapping region
        ict = np.count_nonzero(intersect)  # count intersection area
        if ict <= hullct*tolerance:
            return 0  # the region and the view are not intersecting
        elif ict >= hullct*(1-tolerance):
            return 2  # the region is fully within the view
        else:
            return 1  # the region is partially within the view

    def cubeIntersect(self, cube):
        imsz = (self.imsize[0], self.imsize[1])

        corners = utils.createCubeCorners(cube)  # create 3D corners of the cube from the corner and size
        all_view_pts = self.projectPointsToAllViews(corners)  # project into every view angle

        istatus = []  # intersection status list

        for idx, mask in enumerate(self.masks):
            points = all_view_pts[idx]
            verts = hull.createHullPoints(points)
            _, bhull = hull.drawHull(verts, imsz)
            istatus.append(self.intersectStatus(bhull, mask))

        istatus = np.array(istatus, int)
        if np.any(istatus == 0):
            return 0  # this cube is external
        elif np.all(istatus == 2):
            return 2  # this cube is completely internal
        else:
            return 1  # this cube is partially intersecting

    def octTree(self, cube, depth, limit):
        intersect = self.cubeIntersect(cube)
        if intersect == 2:
            return []

        if intersect == 0:
            return []

        if depth >= limit:
            return [cube]
        else:
            octcubes = utils.cubeOctsect(cube)
            outcubes = []

            for newcube in octcubes:
                outcubes = outcubes + self.octTree(newcube, depth+1, limit)

            return outcubes

    def reconstruct(self, limit):
        if limit == 0:
            return []

        model = self.octTree(self.init_cube, 0, limit)
        self.model = model
        return model

    def refine(self, addLevel):
        if addLevel == 0:
            return self.model

        model = []
        for cube in self.model:
            model = model + self.octTree(cube, 0, addLevel)

        self.model = model
        return model

    def drawModel(self):
        figure = plt.figure()
        fig = figure.add_subplot(111, projection='3d')
        fig.set_aspect('equal')
        for cube in self.model:
            verts = utils.createCubeCorners(cube)
            verts[:, 2] = -verts[:, 2]

            edges = [ [verts[0], verts[1], verts[2], verts[3]],
                      [verts[4], verts[5], verts[6], verts[7]],
                      [verts[0], verts[1], verts[5], verts[4]],
                      [verts[2], verts[3], verts[7], verts[6]],
                      [verts[1], verts[2], verts[6], verts[5]],
                      [verts[0], verts[3], verts[7], verts[4]]
                      ]

            #fig.scatter3D(verts[:, 0], verts[:, 1], verts[:, 2])
            faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
            faces.set_facecolor((0, 0, 1, 1))
            fig.add_collection3d(faces)

        fig.set_xlim(-60, 60)
        fig.set_ylim(-60, 60)
        fig.set_zlim(-20, 100)
        plt.show()

    def rotateModel(self, savename):
        figure = plt.figure()
        fig = figure.add_subplot(111, projection='3d')
        fig.set_aspect('equal')
        for cube in self.model:
            verts = utils.createCubeCorners(cube)
            verts[:, 2] = -verts[:, 2]

            edges = [ [verts[0], verts[1], verts[2], verts[3]],
                      [verts[4], verts[5], verts[6], verts[7]],
                      [verts[0], verts[1], verts[5], verts[4]],
                      [verts[2], verts[3], verts[7], verts[6]],
                      [verts[1], verts[2], verts[6], verts[5]],
                      [verts[0], verts[3], verts[7], verts[4]]
                      ]

            #fig.scatter3D(verts[:, 0], verts[:, 1], verts[:, 2])
            faces = Poly3DCollection(edges, linewidths=0.1, edgecolors='k')
            faces.set_facecolor((0, 0, 1, 1))
            fig.add_collection3d(faces)

        fig.set_xlim(-60, 60)
        fig.set_ylim(-60, 60)
        fig.set_zlim(-20, 100)

        for angle in range(0,360,5):
            fig.view_init(elev=10., azim=angle)
            plt.savefig(savename + "%03d.png" % angle)

    def save(self, fname):
        pickle.dump(self, open(fname+'.obj', 'wb'))



