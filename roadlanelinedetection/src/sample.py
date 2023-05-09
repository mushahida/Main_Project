import cv2
import numpy as np

# Load the image
image = cv2.imread('road.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection using Canny algorithm
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Hough transform to detect lines
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=75, minLineLength=100, maxLineGap=5)

# Draw detected lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    slope = (y2 - y1) / (x2 - x1 + 1e-6)
    print(abs(slope),"===============================")
    if abs(slope)>10:
        print("zig zag line detected")

# Show the image with detected lines
cv2.imshow('Road Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
