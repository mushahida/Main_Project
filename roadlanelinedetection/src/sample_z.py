import cv2
import numpy as np
def zebra_line(path):


    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,
                               (15, 15), 6)

    ret, thresh = cv2.threshold(blurred,
                                180, 255,
                                cv2.THRESH_BINARY)

    _,contours, hier = cv2.findContours(thresh.copy(),
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    count=0
    for c in contours:
        # if the contour is not sufficiently large, ignore it
        if  cv2.contourArea(c) < 2000 or cv2.contourArea(c) > 4000:
            continue
        count=count+1
        # get the min area rect
        print(cv2.contourArea(c),"===============")
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    cv2.imwrite('zebra_lane1.jpg', img)
    print("======================*************",count)
    if count>6:
        print("zebra_lane detected")
        return "zebra lane detected",img
    else:
        return "na", img
