import numpy as np

def loadCameraParameters(fname):
    params = np.load(fname)
    cam_matrix = params['cam_matrix']
    cam_distortion = params['distortion']

    return(cam_matrix, cam_distortion)