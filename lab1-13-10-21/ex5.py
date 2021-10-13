import sys

import cv2.cv2 as cv

imagePath = sys.argv[1]
cascPath = sys.argv[1]  # cascade name

faceCascade = cv.CascadeClassifier(cascPath)

image = cv.imread(imagePath)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)  # ,
    # flags = cv.cv.CV_HAAR_SCALE_IMAGE
)

print('Found {0} faces'.format(len(faces)))

for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow("Faces", image)
cv.waitKey(0)
