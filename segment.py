import numpy as np
import cv2
import matplotlib.pyplot as plt


def segment(im_orig):
    im = im_orig[300:1100, 100:700, :]

    # show_im = np.transpose(cv2.resize(im, (int(im.shape[1] / 2), int(im.shape[0] / 2))), (0, 1, 2))
    # r = cv2.selectROI('im', show_im, False, False)
    # r = tuple([2*p for p in r])
    # print(r)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # r = (112, 244, 632, 904)  #x,y,w,h

    im = k_means_segment(im, 8)

    r = (10, 10, im.shape[1]-20, im.shape[0]-20)
    im_seg = grab_cut_segment(im, r)

    full_im_seg = np.zeros(im_orig.shape)
    full_im_seg[300:1100, 100:700, :] = im_seg

    mask = np.zeros((full_im_seg.shape[:2]))
    mask[np.logical_or(
        np.logical_or(full_im_seg[:, :, 0] > 0,
                        full_im_seg[:, :, 1] > 0),
                         full_im_seg[:, :, 2] > 0)] = 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask = mask > 0
    return mask


def grab_cut_segment(im, bounds):
    mask = np.zeros(im.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(im, mask, bounds, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    im = im * fg[:, :, np.newaxis]
    return im


def k_means_segment(im, k):
    pixels = im.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((im.shape))


if __name__ == '__main__':
    for i in ['01', '07', '10', '11', '14', '32']:
        im_name = './rubiks/rubiks{}.jpg'.format(i)
        im_real = cv2.imread(im_name)
        im_seg = segment(im_real)
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(im_real, cv2.COLOR_BGR2RGB))
        plt.subplot(122)
        plt.imshow(im_seg, cmap='Greys_r')
        # plt.savefig('fig{}'.format(i))
        plt.show()
