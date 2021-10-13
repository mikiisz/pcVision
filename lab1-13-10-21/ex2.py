import matplotlib.pyplot as plt

from skimage import data, filters

image = data.coins()

plt.imshow(image, cmap='gray')
print('image.shape={} image.max={}'.format(image.shape, image.max()))
plt.show()

edges = filters.sobel(image)

plt.imshow(edges, cmap='gray')
plt.show()
