import imp
import cv2
import matplotlib.image as img


def read(image_path):
    image = cv2.imread(image_path)
    b, g, r = cv2.split(image)
    # return cv2.merge([r, g, b])
    return image


def write(image_path, image):
    img.imsave(image_path, image, cmap="gray")
