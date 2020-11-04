# by Julie McGlensey
# This script:
#   processes a set of images from the NIST database of fingerprint images to extract features,
#   compares a reference image against all of the images in the TEST folder,
#   removes the reference image from the TEST folder and repeat until the TRAIN folder has been
#       entirely processed.

from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as ssim
import shutil
import os

# paths
train_path = Path(r'F:\auth\NISTSpecialDatabase4GrayScaleImagesofFIGS\sd04\TRAIN')
test_path = Path(r'F:\auth\NISTSpecialDatabase4GrayScaleImagesofFIGS\sd04\test')
test_dump_path = Path(r'F:\auth\NISTSpecialDatabase4GrayScaleImagesofFIGS\sd04\test_dump')


# preprocessing
def cleanup_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    return cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 181, 11)


# check if images are (actually) the same print
def check(entry1, entry2):
    match = 0
    name1 = str(entry1.name)[1:5]
    name2 = str(entry2.name)[1:5]
    if name1 == name2:
        match = 1
    return match


# compare two images for structural similarities
def compare(image1, image2, match):
    s = ssim(image1, image2)
    if s >= 0.083 and match == 0:
        # false accept
        return 0
    if s < 0.083 and match == 1:
        # false reject
        return 1
    if s >= 0.083 and match == 1:
        # true accept
        return 2
    if s < 0.083 and match == 0:
        # true reject
        return 3


# calculate accuracy of the system
def accuracy(results):
    count = results.count(2) + results.count(3)
    r = count / len(results)
    return r


# calculate false acceptance rate (FAR)
def far(results):
    count = results.count(0)
    r = count / len(results)
    return r


# calculate false rejection rate (FRR)
def frr(results):
    count = results.count(1)
    r = count / len(results)
    return r


# process images
def train():
    results = []
    for entry1 in test_path.iterdir():
        f_print1 = cleanup_img(entry1)
        if len(os.listdir(test_dump_path)) != 0:
            for entry2 in test_dump_path.iterdir():
                f_print2 = cleanup_img(entry2)
                match = check(entry1, entry2)
                result = compare(f_print1, f_print2, match)
                results.append(result)
            shutil.move(str(entry1), test_dump_path)
        else:
            shutil.move(str(entry1), test_dump_path)
    print("Accuracy: " + str(accuracy(results)))
    print("Average FRR: " + str(frr(results)))
    print("Average FAR: " + str(far(results)))


if __name__ == '__main__':
    train()
