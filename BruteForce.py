# by Shannon McHale 
# Based on: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
#  

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread
from skimage.filters import prewitt_h, prewitt_v
from skimage.metrics import structural_similarity as ssim
import shutil
import os



# variables

train_path = Path(r'F:\auth\NISTSpecialDatabase4GrayScaleImagesofFIGS\sd04\TRAIN')
test_path = Path(r'F:\auth\NISTSpecialDatabase4GrayScaleImagesofFIGS\sd04\TEST')
test_test = Path(r'F:\auth\test')
test_test_train = Path(r'F:\auth\test_train')


# preprocessing? see if this increases accuracy after completion
def cleanup_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    return cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 181, 11)


# calculate the mean squared error between two images
def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])
    return err


# compare two images for structural similarities
def compare(image1, image2, title):
    sift = cv2.xfeatures2d.SIFT_create()
    
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), 
             dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []
   
    for p, q in matches:
        if p.distance < 0.1*q.distance:
            match_points.append(p)
    
    keypoints = 0
    if len(keypoints_1) <= len(keypoints_2):
        keypoints = len(keypoints_1)            
    else:
        keypoints = len(keypoints_2)
    if (len(match_points) / keypoints)>0.95:
        print("% match: ", len(match_points) / keypoints * 100)
        print("Figerprint ID: " + str(file)) 
        result = cv2.drawMatches(test_original, keypoints_1, fingerprint_database_image, 
                            keypoints_2, match_points, None) 
        result = cv2.resize(result, None, fx=2.5, fy=2.5)
    cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break;

# process images
def train():
    for entry1 in test_test.iterdir():
        # f_print1 = imread(str(entry1), as_gray=True)
        f_print1 = cleanup_img(entry1)
        # prewitt_h_edges = prewitt_h(f_print1)
        # prewitt_v_edges = prewitt_v(f_print1)
        if len(os.listdir(test_test_train)) != 0:
            for entry2 in test_test_train.iterdir():
                # f_print2 = imread(str(entry2), as_gray=True)
                f_print2 = cleanup_img(entry2)
                compare(f_print1, f_print2, "comparison")
            shutil.move(str(entry1), test_test_train)
        else:
            shutil.move(str(entry1), test_test_train)


if __name__ == '__main__':
    train()
