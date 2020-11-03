from pathlib import Path
import numpy as np
from PIL import Image
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def get_labels(label_files):
    labels = []
    for i in range(len(label_files)):
        f = label_files[i]
        f = open(f, 'r')
        temp = f.readlines()
        f.close()
        label = temp[1].split(' ')[1].strip()

        if label == 'A':
            label_convert = 0
        elif label == 'L':
            label_convert = 1
        elif label == 'R':
            label_convert = 2
        elif label == 'T':
            label_convert = 3
        else:
            label_convert = 4

        labels.append(label_convert)
    return labels


def get_images(image_files):
    images = []
    for i in range(len(image_files)):
        image_file = image_files[i]
        image = Image.open(image_file)
        data = np.asarray(image)
        images.append(data)
        image.close()

    return images


def main():
    # This should be the path where all the txt files and image files are
    data_path = Path(
        r"C:\Users\kysmi\Desktop\CSEC 472\Labs\4\NISTSpecialDatabase4GrayScaleImagesofFIGS\sd04\data")

    # Grab all the files
    all_image_files = list(data_path.glob("*.png"))
    all_label_files = list(data_path.glob("*.txt"))

    # Get the files and place them into numpy arrays
    train_labels = np.array(get_labels(all_label_files))
    test_labels = np.array(get_labels(all_label_files))

    train_images = np.array(get_images(all_image_files))
    test_images = np.array(get_images(all_image_files))

    # Reduce the arrays so the values are between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape the arrays
    nsamples, nx, ny = train_images.shape
    train_images2 = train_images.reshape((nsamples, nx*ny))

    nsamples, nx, ny = test_images.shape
    test_images2 = test_images.reshape((nsamples, nx * ny))

    # Logistic Regression
    regression = linear_model.LogisticRegression()
    regression.fit(train_images2, train_labels)
    prediction_labels = regression.predict(test_images2)

    correct = 0
    incorrect = 0
    for i in range(len(test_labels)):
        if test_labels[i] == prediction_labels[i]:
            correct += 1
        else:
            incorrect += 1

    fpr, tpr, thresholds = roc_curve(test_labels, prediction_labels, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print("Log Regression Stats:")
    print("Number Correct: ", correct)
    print("Number Incorrect: ", incorrect)
    print("Average Correct: ", correct/(correct+incorrect))
    print("Average Incorrect: ", incorrect/(correct+incorrect))
    print("Equal Error Rate: ", eer)

    # Support Vector Machine
    clf = svm.SVC()
    clf.fit(train_images2, train_labels)
    prediction_labels = clf.predict(test_images2)

    correct = 0
    incorrect = 0
    for i in range(len(test_labels)):
        if test_labels[i] == prediction_labels[i]:
            correct += 1
        else:
            incorrect += 1

    fpr, tpr, thresholds = roc_curve(test_labels, prediction_labels, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print("SVM Stats:")
    print("Number Correct: ", correct)
    print("Number Incorrect: ", incorrect)
    print("Average Correct: ", correct / (correct + incorrect))
    print("Average Incorrect: ", incorrect / (correct + incorrect))
    print("Equal Error Rate: ", eer)

    # Naive Bayes
    gnb = GaussianNB()
    gnb.fit(train_images2, train_labels)
    prediction_labels = gnb.predict(test_images2)

    correct = 0
    incorrect = 0
    for i in range(len(test_labels)):
        if test_labels[i] == prediction_labels[i]:
            correct += 1
        else:
            incorrect += 1

    fpr, tpr, thresholds = roc_curve(test_labels, prediction_labels, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print("Naive Bayes Stats:")
    print("Number Correct: ", correct)
    print("Number Incorrect: ", incorrect)
    print("Average Correct: ", (correct / (correct + incorrect)))
    print("Average Incorrect: ", (incorrect / (correct + incorrect)))
    print("Equal Error Rate: ", eer)


main()
