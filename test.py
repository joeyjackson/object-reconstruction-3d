from pointProjection import Reconstructor
import utils


Recon = Reconstructor(utils.loadCameraParameters(), 10, utils.getImageStack('rubiks'))
print(Recon.projectPointsToAllViews([[1,2,3]]))


print(utils.createCubeCorners([-30, -30, 0], 60))