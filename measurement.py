import cv2
import utils

# Name of the images to save as
samplename = 'rubiks2/rubiks2'

# Begin video capture
cv2.namedWindow("Press ENTER to take measurement, ESC to exit.\n" +
                "Rotate the turntable by a fixed angle for each  capture.")
vc = cv2.VideoCapture(1)
vc.set(3, 1920)
vc.set(4, 1080)

if vc.isOpened():
    ret, img = vc.read()
else:
    ret = False

# Begin save loop
sample = 1
while ret:

    cv2.imshow("Press ENTER to take measurement, ESC to exit.\n" +
               "Rotate the turntable by a fixed angle for each capture.",
               cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)) ) )  # show camera feed scaled by half
    ret, img = vc.read()
    img = utils.flipImage(img)

    key = cv2.waitKey(20)
    if key == 13:  # user hit enter
        rval = cv2.imwrite(samplename+"%02d"%sample+".jpg", img)
        print("Took measurement " + str(sample))
        sample += 1
    if key == 27:  # user hit esc
        break

cv2.destroyWindow("Preview")