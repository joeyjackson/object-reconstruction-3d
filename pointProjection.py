import cv2
import numpy as np
import utils


class Reconstructor:

    def __init__(self, params, imgs):
        self.cam_mtx = params[0]  # camera matrix
        self.cam_dist = params[1]  # camera distortion
        self.cam_pos = params[2]  # camera position
        self.cam_rot = params[3]  # camera rotation
        self.masks = []  # binary foreground masks

        self.cubes = []  # a list of cubes corner points. may not be needed