import numpy as np
import os
import shutil

def dog_training_moving():
    if not os.path.exists("./train/dogs"):
        os.mkdir("./train/dogs")
    for i in range(1000):
        origin_image_path = os.path.join("./all/", "dog."+str(i)+".jpg")
        target_image_path = os.path.join("./train/dogs/", "dog_"+str(i) + ".jpg")
        shutil.copy(origin_image_path, target_image_path)

def cat_training_moving():
    if not os.path.exists("./train/cats"):
        os.mkdir("./train/cats")
    for i in range(1000):
        origin_image_path = os.path.join("./all/", "cat."+str(i)+".jpg")
        target_image_path = os.path.join("./train/cats/", "cat_"+str(i) + ".jpg")
        shutil.copy(origin_image_path, target_image_path)

def dog_validation_moving():
    if not os.path.exists("./validation/dogs"):
        os.mkdir("./validation/dogs")
    for i in range(1000, 1400):
        origin_image_path = os.path.join("./all/", "dog."+str(i)+".jpg")
        target_image_path = os.path.join("./validation/dogs/", "dog_"+str(i) + ".jpg")
        shutil.copy(origin_image_path, target_image_path)

def cat_validation_moving():
    if not os.path.exists("./validation/cats"):
        os.mkdir("./validation/cats")
    for i in range(1000, 1400):
        origin_image_path = os.path.join("./all/", "cat."+str(i)+".jpg")
        target_image_path = os.path.join("./validation/cats/", "cat_"+str(i) + ".jpg")
        shutil.copy(origin_image_path, target_image_path)


if __name__ == "__main__":
    dog_training_moving()
    cat_training_moving()
    dog_validation_moving()
    cat_validation_moving()