import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from timeit import default_timer as timer
from tqdm import tqdm

# Please define additional libraries here if needed

LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def preprocess_image(path_to_image, img_size=150):
    """
    Read and resize an input image
    :param path_to_image: path of image file
    :param img_size: image size
    :return: image as a Numpy array
    """
    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size))
    return np.array(img)


def extract_color_histogram(dataset, hist_size=6):
    """
    Extract colour histogram features from a dataset of images
    :param dataset: dataset of images
    :param hist_size: histogram size for each dimension
    :return: colour histograms
    """
    col_hist = []
    for img in dataset:
        hist = cv2.calcHist([img], [0, 1, 2], None, (hist_size, hist_size, hist_size), [0, 256, 0, 256, 0, 256])
        col_hist.append(cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX).flatten())
    return np.array(col_hist)


def load_dataset(base_path='flowers'):
    X = []
    Y = []
    for i in range(0, len(LABELS)):
        current_size = len(X)
        for img in tqdm(os.listdir(base_path + os.sep + LABELS[i])):
            X.append(preprocess_image(base_path + os.sep + LABELS[i] + '/' + img))
            Y.append(LABELS[i])
        print(f'Loaded {len(X) - current_size} {LABELS[i]} images')
    return X, Y


# Please define your own functions here if needed


if __name__ == '__main__':
    # 1. Load dataset
    X, Y = load_dataset()

    # Please insert your own code here to deal with the tasks
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.6, test_size=0.2,
                                                        shuffle=True, random_state=42)
