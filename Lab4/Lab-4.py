from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# Paths of images
train = Path(r'C:\Users\kysmi\Desktop\CSEC 472\Labs\4\Data\train')
test = Path(r'C:\Users\kysmi\Desktop\CSEC 472\Labs\4\Data\test')
data_test = Path(r'C:\Users\kysmi\Desktop\CSEC 472\Labs\4\Data\data_test')
data_train = Path(r'C:\Users\kysmi\Desktop\CSEC 472\Labs\4\Data\data_train')

# Path to train and test folders
test_image_path = Path(r'C:\Users\kysmi\Desktop\CSEC 472\Labs\4\test')
train_image_path = Path(r'C:\Users\kysmi\Desktop\CSEC 472\Labs\4\train')


# Function to determine type of minutiae at pixel P(i,j) ....

def minutiae_at(pixels, i, j):
    values = [pixels[i + k][j + l] for k, l in cells]

    crossings = 0
    for k in range(0, 8):
        crossings += abs(values[k] - values[k + 1])
    crossings /= 2

    if pixels[i][j] == 1:
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"
    return "none"


# Function to convert the image into pixels ....

def load_image(im):
    (x, y) = im.size
    im_load = im.load()

    result = []
    for i in range(0, x):
        result.append([])
        for j in range(0, y):
            result[i].append(im_load[i, j])

    return result


# Function to apply particular property to each pixel ....

def apply_to_each_pixel(pixels, f):
    for i in range(0, len(pixels)):
        for j in range(0, len(pixels[i])):
            pixels[i][j] = f(pixels[i][j])


# Function to show minutiae on the image ....

def show_minutiaes(im):
    pixels = load_image(im)
    apply_to_each_pixel(pixels, lambda x: 0.0 if x > 10 else 1.0)

    (x, y) = im.size
    result = im.convert("RGB")

    draw = ImageDraw.Draw(result)

    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}

    ellipse_size = 8
    for i in range(1, x - 1):
        for j in range(1, y - 1):
            minutiae = minutiae_at(pixels, i, j)
            if minutiae != "none":
                draw.ellipse([(i - ellipse_size, j - ellipse_size), (i + ellipse_size, j + ellipse_size)],
                             outline=colors[minutiae])

    del draw
    return result


def extract_minutae(image_path, train):
    count = 1
    for path in image_path:
        Img_Original = io.imread(path)
        Otsu_Threshold = threshold_otsu(Img_Original)
        BW_Original = Img_Original > Otsu_Threshold
        BW_Skeleton = skeletonize(BW_Original)
        imgplot = plt.imshow(BW_Skeleton)
        imgplot.set_cmap('hot')
        plt.axis('off')
        save_file_path = 'skeleton/' + str(count) + '_skeleton.png'
        plt.savefig(save_file_path)
        plt.close()

        image = Image.open(save_file_path).convert('L')
        # image.save('thinned_greyscale.png')

        # Img_Original = io.imread('thinned_greyscale.png')
        Minutiae_Image = show_minutiaes(image)

        # Displaying the results ....
        # fig, ax = plt.subplots(1, 2)
        # ax1, ax2 = ax.ravel()
        # ax1.imshow(image, cmap=plt.cm.gray)
        # ax1.set_title('Original image')
        # ax1.axis('off')
        # ax2.imshow(Minutiae_Image, cmap=plt.cm.gray)
        # ax2.set_title('Minutiae in the image')
        # ax2.axis('off')
        # plt.show()
        # plt.close()

        # Saving the image ....
        if train:
            save_file_path = 'train/' + str(count) + '.png'
        else:
            save_file_path = 'test/' + str(count) + '.png'
        Minutiae_Image.save(save_file_path)
        count += 1


def get_images(image_files):
    images = []
    for i in range(len(image_files)):
        image_file = image_files[i]
        image = Image.open(image_file)
        data = np.asarray(image)
        images.append(data)
        image.close()

    return images


def display_image(train_images):
    plt.figure(figsize=(10, 10))
    for i in range(len(train_images)):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.show()


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def main():
    train_images = list(train.glob("*.png"))
    test_images = list(data_test.glob("*.png"))

    print("Normal picture comparison")
    train_img = np.array(get_images(train_images))
    test_img = np.array(get_images(test_images))
    train_img = train_img / 255.0
    test_img = test_img / 255.0

    image_count = 1
    input_count = 1
    for image in test_img:
        for image2 in train_img:
            mse_ = mse(image, image2)
            print("MSE for " + str(image_count) + " and " + str(input_count) + ": " + str(mse_))
            if mse_ == 0:
                print("Found a match between image " + str(image_count) + " and " + str(input_count))
            input_count += 1
        input_count = 1
        image_count += 1

    print("Minutae picture comparison")
    extract_minutae(train_images, True)
    extract_minutae(test_images, False)

    training_img = list(train_image_path.glob("*.png"))
    testing_img = list(test_image_path.glob("*.png"))

    training_img = np.array(get_images(training_img))
    testing_img = np.array(get_images(testing_img))

    training_img = training_img / 255.0
    testing_img = testing_img / 255.0

    # display_image(training_img)
    # display_image(testing_img)

    image_count = 1
    input_count = 1
    for image in testing_img:
        for image2 in training_img:
            mse_ = mse(image, image2)
            print("MSE for " + str(image_count) + " and " + str(input_count) + ": " + str(mse_))
            if mse_ == 0:
                print("Found a match between image " + str(image_count) + " and " + str(input_count))
            input_count += 1
        input_count = 1
        image_count += 1


main()
