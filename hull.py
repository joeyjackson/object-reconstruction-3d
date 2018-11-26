import numpy as np
import cv2


def createHullPoints(pts):
    # Returns the vertices of a convex hull around the given points
    #
    # ---- Inputs ----------------------------
    # pts = Nx2 numpy array of 2D points. array[[x,y],[x,y]...]
    #
    # ----- Outputs ---------------------------
    # verts = Mx2 numpy array of 2D points which are the vertices of the convex hull around the input points
    #
    #
    inpts = np.array(pts, np.float32)  # convert to floating point numpy array if not already
    verts = cv2.convexHull(inpts)

    verts = verts[:, 0]  # For some reason convex hull puts out a 3D np array, reduce to 2

    return verts


def drawHull(verts, size):
    # Returns an image with the convex hull bound by verts filled in
    #
    # ---- Inputs ----------------------------
    # verts = Nx2 numpy array of 2D points. array[[x,y],[x,y]...]
    # size = length 2 tuple or list with the height and width of the resulting image specified (h, w)
    #
    # ----- Outputs ---------------------------
    # img = Color image of size (size[0], size[1], 3) with everything in black and the hull filled white
    # binimg = Binary image of size (size) with everything false and the hull filled in True
    #
    idxs = np.array(np.round(verts), int)  # round points into indices of the convex hull
    vertices = np.array([idxs])  # vertices of the convex hull, 3D array (2D wrapped once)

    img = np.zeros((size[0], size[1], 3), np.uint8)
    img = cv2.drawContours(img, vertices, -1, thickness=cv2.FILLED, color=(255, 255, 255))

    binimg = np.array(img, np.bool)[:, :, 0]

    return img, binimg