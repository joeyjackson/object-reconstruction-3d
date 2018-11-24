import cv2

cv2.namedWindow("view")
vc = cv2.VideoCapture(1)
vc.set(3,1920)
vc.set(4,1080)

if vc.isOpened():
    ret, img = vc.read()
else:
    ret = False


fname = "Calibrator"
sample = 11
while ret:
    cv2.imshow("view", img)
    ret, img = vc.read()
    key = cv2.waitKey(20)
    if key == 13:
        rval = cv2.imwrite(fname+str(sample)+".jpg",img)
        print("Took calibrator measurement " + str(sample))
        sample+=1
    if key == 27:
        break

cv2.destroyWindow("Preview")