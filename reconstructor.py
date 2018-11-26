import cv2
import utils
import segment
import hull
import numpy as np

class Reconstructor:

    def __init__(self, params, angle, imgstack):
        self.cam_mtx = params[0]  # camera matrix
        self.cam_dist = params[1]  # camera distortion
        self.cam_pos = params[2]  # camera position
        self.cam_rot = params[3]  # camera rotation

        self.increment = angle  # rotation per captured image in degrees

        self.masks = []  # binary foreground masks
        self.imsize = imgstack[0].shape
        for img in imgstack:
            self.masks.append(None)
            #self.masks.append(segment.segment(img))  # get binary silhouette of foreground

        self.cubes = []  # a list of cubes corner points. may not be needed

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

    def intersectStatus(self, bhull, bview, tolerance=0.02):
        hullct = np.count_nonzero(bhull)  # count hull area

        intersect = np.logical_and(bhull,bview)  # overlapping region
        ict = np.count_nonzero(intersect)  # count intersection area
        if ict <= hullct*tolerance:
            return 0  # the region and the view are not intersecting
        elif ict >= hullct*(1-tolerance):
            return 2  # the region is fully within the view
        else:
            return 1  # the region is partially within the view

    def cubeIntersect(self, cube):
        imsz = (self.imsize[0], self.imsize[1])

        corners = utils.createCubeCorners(cube[0], cube[1])  # create 3D corners of the cube from the corner and size
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
        else
            return 1  # this cube is partially intersecting





