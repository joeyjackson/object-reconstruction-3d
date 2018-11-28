from reconstructor import Reconstructor
import utils
import pickle


def main():
    calfile = 'Camera_Calibration.npz'
    imfolder = 'spray'
    angle = 10
    initcube = ((-50, -50, 0), 100)
    resolution = 8

    save = True
    savename = 'obj/spray_resolution'

    load = False

    if load:
        Recon = pickle.load(open(savename + '.obj', 'rb'))
        Recon.refine(2)
    else:
        Recon = Reconstructor(utils.loadCameraParameters(calfile), angle, utils.getImageStack(imfolder), initcube)
        Recon.reconstruct(resolution)

    Recon.drawModel()
    if save:
        Recon.save(savename)

if __name__ == '__main__':
    main()
