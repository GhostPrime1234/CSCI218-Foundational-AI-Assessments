"""
Implement an MLP classifier for image classification using Scikit Learn
@author: Thanh Le
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from timeit import default_timer as timer
from tqdm import tqdm

LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def preprocess_image(path_to_image, img_size=256):
    """Read and resize an input image"""
    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size))
    return np.array(img)


def extract_color_histogram(dataset, hist_size=6):
    """Extract color histogram features from a dataset of images"""
    col_hist = []
    for img in dataset:
        hist = cv2.calcHist([img], [0, 1, 2], None, (hist_size, hist_size, hist_size), [0, 256, 0, 256, 0, 256])
        col_hist.append(cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX).flatten())
    return np.array(col_hist)


def load_dataset(base_path='flowers'):
    """Load dataset from the specified directory"""
    X, Y = [], []
    for label in LABELS:
        current_size = len(X)
        for img in tqdm(os.listdir(os.path.join(base_path, label)), desc=f"Loading {label} "):
            X.append(preprocess_image(os.path.join(base_path, label, img)))
            Y.append(label)
        # print(f'Loaded {len(X) - current_size} {label} images')
    return X, Y


def display_sample_images(dataset, labels, predictions, correct_indices, incorrect_indices, num_samples=5):
    """Display sample images with correct and incorrect classifications"""
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    for i in range(num_samples):
        correct_idx = correct_indices[i]
        incorrect_idx = incorrect_indices[i]

        axes[i, 0].imshow(cv2.cvtColor(dataset[correct_idx], cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"True: {labels[correct_idx]} | Predicted: {predictions[correct_idx]}")
        axes[i, 0].set_axis_off()

        axes[i, 1].imshow(cv2.cvtColor(dataset[incorrect_idx], cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"True: {labels[incorrect_idx]} | Predicted: {predictions[incorrect_idx]}")
        axes[i, 1].set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    start_time = timer()

    # STEP 1. Load dataset
    X, y = load_dataset()

    # STEP 2. Split dataset into train, validation, and test datasets
    train_data, valTest_data, y_train, valTest_labels = train_test_split(X, y, test_size=0.4, random_state=42,
                                                                         shuffle=True)
    valid_data, test_data, valid_labels, y_test = train_test_split(valTest_data, valTest_labels, test_size=0.5,
                                                                   random_state=42, shuffle=True)

    # STEP 3. Extract colour histogram features from the datasets
    X_train = extract_color_histogram(train_data)
    X_valid = extract_color_histogram(valid_data)
    X_test = extract_color_histogram(test_data)

    # STEP 4. Define 9 different structures
    n_hidden_1 = int(np.random.randint(5, 6 ** 3))
    n_hidden_2 = int(2 / 3 * (6 ** 3) + 5)
    n_hidden_3 = int(np.random.randint(0, 2 * (6 ** 3)))

    n_hidden_options = [
        n_hidden_1,
        n_hidden_2,
        n_hidden_3,
        (n_hidden_1, n_hidden_1),
        (n_hidden_2, n_hidden_2),
        (n_hidden_3, n_hidden_3),
        (n_hidden_1, n_hidden_1, n_hidden_1),
        (n_hidden_2, n_hidden_2, n_hidden_2),
        (n_hidden_3, n_hidden_3, n_hidden_3)
    ]

    best_accuracy = 0
    best_structure = None

    # STEP 5. Determine the optimal structure
    for test_case in tqdm(n_hidden_options):
        clf = MLPClassifier(hidden_layer_sizes=test_case, activation='relu', solver='adam', max_iter=1500,
                            random_state=42, early_stopping=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        accuracy = accuracy_score(valid_labels, y_pred) * 100
        print(f"\nStructure: {test_case}, accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_structure = test_case

    print(f"Best accuracy is {best_accuracy:.4f}, best structure is {best_structure}")

    # STEP 6. Train an MLP classifier with the optimal structure
    clf = MLPClassifier(hidden_layer_sizes=best_structure, activation='relu', solver='adam', max_iter=1500,
                        random_state=42, early_stopping=True)
    start_time = timer()
    clf.fit(X_test, y_test)

    training_time = timer() - start_time
    print(f"Training time is {training_time:.4f} seconds")
    # STEP 7. Evaluate the MLP on the test dataset
    start_time = timer()
    y_pred = clf.predict(X_test)
    inference_time = timer() - start_time
    print(f"Inference time is {inference_time:.4f} seconds")

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='macro') * 100
    recall = recall_score(y_test, y_pred, average='macro') * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100

    print(f"Accuracy score: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # STEP 8. Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # STEP 9. Show 5 correctly/incorrectly classified images
    correct_indices = [index for index in range(len(y_test)) if y_pred[index] == y_test[index]]
    incorrect_indices = [index for index in range(len(y_test)) if y_pred[index] != y_test[index]]

    correct_sample_indices = np.random.choice(correct_indices, 5)
    incorrect_sample_indices = np.random.choice(incorrect_indices, 5)

    display_sample_images(test_data, y_test, y_pred, correct_sample_indices, incorrect_sample_indices, num_samples=5)

    end_time = timer()
    print(f"Time taken: {end_time - start_time:.4f} seconds")
