from PIL import Image, ImageFilter

im = Image.open('messi5.jpg')
im.show()

im_sharp = im.filter(ImageFilter.SHARPEN)
im_sharp.save('messi5_sharpen.jpg', 'JPEG')

r, g, b = im_sharp.split()

exif_data = im._getexif()
exif_data
