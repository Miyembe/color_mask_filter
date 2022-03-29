import cv2

def doNothing(x):
    pass

cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, doNothing)
cv2.createTrackbar('min_green', 'Track Bars', 0, 255, doNothing)
cv2.createTrackbar('min_red', 'Track Bars', 0, 255, doNothing)
cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, doNothing)
cv2.createTrackbar('max_green', 'Track Bars', 0, 255, doNothing)
cv2.createTrackbar('max_red', 'Track Bars', 0, 255, doNothing)

object_image = cv2.imread(image_path)
resized_image = cv2.resize(object_image, (800, 626))
hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

cv2.imshow('Base Image', resized_image)
cv2.imshow('HSV Image', hsv_image)

