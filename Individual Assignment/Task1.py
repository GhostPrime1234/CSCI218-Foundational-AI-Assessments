import os
import timeit

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from tqdm import tqdm

LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def preprocess_image(path_to_image, img_size=150) -> np.ndarray:
    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size))
    return np.array(img)


def extract_color_histogram(dataset, hist_size=3) -> np.ndarray:
    col_hist = []
    for img in dataset:
        hist = cv2.calcHist([img], [0, 1, 2], None, (hist_size, hist_size, hist_size), [0, 256, 0, 256, 0, 256])
        col_hist.append(cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX).flatten())
    return np.array(col_hist)


def load_dataset(base_path='flowers') -> tuple[list, list]:
    X = []
    Y = []
    for i in range(0, len(LABELS)):
        # current_size = len(X)
        for img in tqdm(os.listdir(base_path + os.sep + LABELS[i]), desc=f"Loading {LABELS[i]}"):
            X.append(preprocess_image(base_path + os.sep + LABELS[i] + '/' + img))
            Y.append(LABELS[i])
        # print(f'Loaded {len(X) - current_size} {LABELS[i]} images')
    return X, Y


def display_sample_images(m_dataset, m_labels, m_predictions, m_correct_indices, m_incorrect_indices, m_num_samples=5):
    fig, axes = plt.subplots(m_num_samples, 2, figsize=(10, m_num_samples * 5))
    for i in range(m_num_samples):
        correct_idx = m_correct_indices[i]
        incorrect_idx = m_incorrect_indices[i]

        axes[i, 0].imshow(cv2.cvtColor(m_dataset[correct_idx], cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"True: {m_predictions[correct_idx]}")
        axes[i, 0].set_axis_off()

        axes[i, 1].imshow(cv2.cvtColor(m_dataset[incorrect_idx], cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"True: {m_labels[incorrect_idx]} | Predicted: {m_predictions[incorrect_idx]}")
        axes[i, 1].set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    # STEP 1. Load dataset
    X, y = load_dataset()

    # STEP 2. Split dataset into train, validation, and test
    train_data, valTest_data, y_train, valTest_labels = train_test_split(X, y, test_size=0.4, random_state=42,
                                                                         shuffle=True)
    valid_data, test_data, valid_labels, y_test = train_test_split(valTest_data, valTest_labels, test_size=0.5,
                                                                   random_state=42, shuffle=True)

    # STEP 3. Extract colour histogram features from the datasets
    X_train = extract_color_histogram(train_data)
    X_valid = extract_color_histogram(valid_data)
    X_test = extract_color_histogram(test_data)

    # STEP 4. Determine the optimal values of k
    k_list = np.arange(1, 31, 1)

    best_accuracy = 0
    best_k = 0

    # Measure the average inference time
    for k in k_list:
        start_time = timeit.default_timer()
        # Fit a kNN classifier from the training set
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Evaluate the classifier on the validation set
        y_pred = knn.predict(X_valid)
        accuracy = accuracy_score(valid_labels, y_pred) * 100
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
        print(f'k={k}, Accuracy={accuracy:.2f}')
    print(f"\nBest accuracy is {best_accuracy:.2f}, k={best_k}")

    # STEP 5. Train a k-NN classifier with the optimal k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    # STEP 6. Evaluate the classifier on the test set
    y_pred = knn.predict(X_test)
    error = (1 - accuracy_score(y_test, y_pred)) * 100
    print(f'k = {best_k}: Error = {error:.2f}\n')

    # Q4. Measure the average inference time
    inference_times = []

    # Sample test data
    sample = X_test[0].reshape(1, -1)
    counter = 1

    # Run the inference 10 times
    while counter <= 10:
        start_time = timeit.default_timer()
        knn.predict(sample)
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time
        inference_times.append(elapsed_time)
        print(f"Inference time {counter} - {elapsed_time}")
        counter += 1

    # Compute the average inference time
    average_inference_time = np.mean(inference_times)
    print(f'Average inference time: {average_inference_time:.4f} seconds\n')

    # STEP 7. Report the classification metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")

    # STEP 8. Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # STEP 9. Show 5 correctly/incorrectly classified images
    correct_indices = [index for index in range(len(y_test)) if y_pred[index] == y_test[index]]
    incorrect_indices = [index for index in range(len(y_test)) if y_pred[index] != y_test[index]]

    # Ensure there are enough samples to show
    num_correct = min(5, len(correct_indices))
    num_incorrect = min(5, len(incorrect_indices))

    correct_sample_indices = np.random.choice(correct_indices, num_correct, replace=False)
    incorrect_sample_indices = np.random.choice(incorrect_indices, num_incorrect, replace=False)

    display_sample_images(test_data, y_test, y_pred, correct_sample_indices, incorrect_sample_indices, m_num_samples=5)
