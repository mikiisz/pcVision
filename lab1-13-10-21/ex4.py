import cv2.cv2 as cv
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

img = cv.imcount('messi5.jpg', 0)

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

plt.savefig('foo.png')
plt.savefig('foo.pdf')
