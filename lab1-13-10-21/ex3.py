import cv2.cv2 as cv

img = cv.imread('messi5.jpg', 0)

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
