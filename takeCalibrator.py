import cv2
import utils

cv2.namedWindow("view")
vc = cv2.VideoCapture(0)
vc.set(3, 1920)
vc.set(4, 1080)

if vc.isOpened():
    ret, img = vc.read()
else:
    ret = False


fname = "Calibrator"
sample = 1
while ret:
    cv2.imshow("view", cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2))))
    ret, img = vc.read()
    img = utils.flipImage(img)

    key = cv2.waitKey(20)
    if key == 13:
        rval = cv2.imwrite(fname+"%02d"%sample+".jpg",img)
        print("Took calibrator measurement " + str(sample))
        sample += 1
    if key == 27:
        break

cv2.destroyWindow("Preview")