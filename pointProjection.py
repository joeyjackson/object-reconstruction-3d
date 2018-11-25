import cv2
import numpy as np
import utils


class Reconstructor:

    def __init__(self, params, angle, imgstack):
        self.cam_mtx = params[0]  # camera matrix
        self.cam_dist = params[1]  # camera distortion
        self.cam_pos = params[2]  # camera position
        self.cam_rot = params[3]  # camera rotation

        self.increment = angle  # rotation per captured image in degrees

        self.masks = []  # binary foreground masks
        for img in imgstack:
            self.masks.append(self.maskForeground(img))  # get binary silhouette of foreground

        self.cubes = []  # a list of cubes corner points. may not be needed

    def maskForeground(self, image):
        # Function to be completed
        binaryImage = None
        return binaryImage

    def projectPoints(self, points, angle):

        rotated_points = utils.rotatePointsAboutZ(angle, points)
        impts, _ = cv2.projectPoints(rotated_points, self.cam_rot, self.cam_pos, self.cam_mtx, self.cam_dist)

        return impts

    def projectPointsToAllViews(self, points):

        viewpts = []
        for idx in range(0, len(self.masks)):
            angle = self.increment * idx
            viewpts.append(self.projectPoints(points, angle))

        return viewpts
