from reconstructor import Reconstructor
import utils
import pickle
import imageio

def main():
    calfile = 'Camera_Calibration.npz'
    savename = 'spray'
    imfolder = 'captures/' + savename + 'b'
    tmpf = 'temp6'
    angle = 10
    initcube = ((-50, -50, 0), 100)

    print(savename)
    print(tmpf)


    for res in range(0,13):
        if res == 0:
            Recon = Reconstructor(utils.loadCameraParameters(calfile), angle, utils.getImageStack(imfolder), initcube)
            Recon.reconstruct(1)
            print('init done')
        else:
            Recon = pickle.load(open('obj/' + savename + str(res - 1) + '.obj', 'rb'))
            Recon.refine(1)

        Recon.save('obj/' + savename + str(res))
        Recon.rotateModel(tmpf + '/' + savename)
        imgs = utils.getImageStack(tmpf)
        imageio.mimsave(savename + str(res) + '.gif', imgs, duration=0.06)


if __name__ == '__main__':
    main()
