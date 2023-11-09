import cv2
import  numpy as np
import os
import matplotlib.pyplot as plt
from blur import blur as bl


def read_image(image_path):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def normalize_image(image_path, image_size):
    image = read_image(image_path)
    image =cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    return image


def write_image(image, image_path):
    cv2.imwrite(image_path, image)






real_images_dir = "D:\\backup\\images_stunmaster\\3pd_align\\"
blur_images_dir = "D:\\backup\\images_stunmaster\\3pd_blur\\"

def blurize_dataset(real_images_dir, blur_images_dir):

    if not os.path.isdir(blur_images_dir):
        os.makedirs(blur_images_dir)


    images_path = [ i for i in os.listdir(real_images_dir)]
    for image_path in images_path:
        img = read_image(real_images_dir + image_path)
        try:
            blur_img = bl.blurring(img)
            write_image(blur_img, blur_images_dir+image_path)
        except:
            pass

# blurize_dataset(real_images_dir, blur_images_dir)




