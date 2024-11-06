import cv2
import mediapipe as mp
import numpy as np
import argparse
import h5py
import math
from collections import deque
import os
import threading
import queue
from warnings import filterwarnings
# from pynput import keyboard
import re

# Create a queue for logging tasks
log_queue = queue.Queue()


def logging_worker():
    while True:
        feature_list, label, dynamic = log_queue.get()
        if feature_list is None:
            break
        if dynamic:
            # Dynamic gestures: (N, T, 88)
            feature_list = np.array(feature_list).reshape(-1, len(feature_list), 88)
        else:
            # Static gestures: (N, 88)
            feature_list = np.array(feature_list).reshape(-1, 88)
        print(f'shape of feature_list: {feature_list.shape}')
        save_hdf5(feature_list, label, dynamic)
        log_queue.task_done()


# Start the logging thread
log_thread = threading.Thread(target=logging_worker)
log_thread.start()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--label", type=str, required=False, help='Label for the data')
    parser.add_argument("--dynamic", action='store_true', help='Dynamic gesture detection')
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images.",
                        default="data/static/asl_dataset")
    args = parser.parse_args()
    # args.label = args.label.capitalize()
    return args


def save_hdf5(sequence, label, dynamic):
    folder = 'dynamic' if dynamic else 'static'
    hdf5_path = f"data/{folder}/{label}.h5"
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, 'a') as f:
        # Extract numeric part of the keys and sort them numerically
        keys = list(f.keys())
        if keys:
            numeric_keys = sorted([int(re.findall(r'\d+', key)[0]) for key in keys])
            next_index = numeric_keys[-1] + 1
        else:
            next_index = 0
        dset_name = f"{label}_{next_index}"
        f.create_dataset(dset_name, data=np.array(sequence))


def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)


def calculate_angle(point1, point2, point3):
    a = np.array([point1.x, point1.y, point1.z])
    b = np.array([point2.x, point2.y, point2.z])
    c = np.array([point3.x, point3.y, point3.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def moving_average(data, window_size):
    return np.mean(list(data)[-window_size:], axis=0)


def main():
    filterwarnings("ignore")
    args = get_args()

    mp_hands = mp.solutions.hands

    # Buffer to store previous frames for smoothing
    buffer_size = 5
    landmark_buffer = deque(maxlen=buffer_size)
    distance_buffer = deque(maxlen=buffer_size)
    angle_buffer = deque(maxlen=buffer_size)

    # Buffer to store sequences for dynamic gesture detection
    sequence_length = 60
    sequence_buffer = deque(maxlen=sequence_length)

    collecting_data = False

    # Load images from teh specified folder
    for root, dirs, files in os.walk(args.folder_path):
        label = os.path.basename(root) # THe label is the directory name (e.g., 'a', 'b', 'c', etc)
        print(f"Processing images for label: {label}")
        for file in files:
            image_path = os.path.join(root, file)
            print(f"Loading image: {image_path}")

            image = cv2.imread(image_path)

            if image is None:
                print(f'Could not read image: {image_path}')
                continue

            image = cv2.flip(image, 1)

            with mp_hands.Hands(
                            static_image_mode=False,
                            max_num_hands=1,
                            model_complexity=1,
                            min_detection_confidence=0.75,
                            min_tracking_confidence=0.75) as hands:

                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
                    print(f"Hand landmarks detected in image: {image_path}")  # Log successful hand detection
                    for hand_landmarks, hand_world_landmarks in zip(results.multi_hand_landmarks,
                                                                    results.multi_hand_world_landmarks):
                        feature_list = []
                        landmark_list = []
                        for landmark in hand_landmarks.landmark:
                            landmark_list.append([landmark.x, landmark.y, landmark.z])
                        landmark_buffer.append(landmark_list)

                        # Log detected hand landmarks
                        print(
                            f"Detected hand landmarks for {label}: {landmark_list[:5]}...")  # Print a preview of the first few landmarks

                        # Calculate distances between key points
                        distances = [
                            calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[4]),
                            calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[8]),
                            calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[12]),
                            calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[16]),
                            calculate_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[20]),
                            calculate_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8]),
                            calculate_distance(hand_landmarks.landmark[8], hand_landmarks.landmark[12]),
                            calculate_distance(hand_landmarks.landmark[12], hand_landmarks.landmark[16]),
                            calculate_distance(hand_landmarks.landmark[16], hand_landmarks.landmark[20]),
                            calculate_distance(hand_landmarks.landmark[5], hand_landmarks.landmark[9]),
                            calculate_distance(hand_landmarks.landmark[9], hand_landmarks.landmark[13]),
                            calculate_distance(hand_landmarks.landmark[13], hand_landmarks.landmark[17])
                        ]
                        distance_buffer.append(distances)

                        # Log distance calculations
                        print(f"Calculated distances for {label}: {distances[:5]}...")  # Preview of first few distances

                        # Calculate angles between joints
                        angles = [
                            calculate_angle(hand_landmarks.landmark[1], hand_landmarks.landmark[2],
                                            hand_landmarks.landmark[3]),
                            calculate_angle(hand_landmarks.landmark[2], hand_landmarks.landmark[3],
                                            hand_landmarks.landmark[4]),
                            calculate_angle(hand_landmarks.landmark[5], hand_landmarks.landmark[6],
                                            hand_landmarks.landmark[7]),
                            calculate_angle(hand_landmarks.landmark[6], hand_landmarks.landmark[7],
                                            hand_landmarks.landmark[8]),
                            calculate_angle(hand_landmarks.landmark[9], hand_landmarks.landmark[10],
                                            hand_landmarks.landmark[11]),
                            calculate_angle(hand_landmarks.landmark[10], hand_landmarks.landmark[11],
                                            hand_landmarks.landmark[12]),
                            calculate_angle(hand_landmarks.landmark[13], hand_landmarks.landmark[14],
                                            hand_landmarks.landmark[15]),
                            calculate_angle(hand_landmarks.landmark[14], hand_landmarks.landmark[15],
                                            hand_landmarks.landmark[16]),
                            calculate_angle(hand_landmarks.landmark[17], hand_landmarks.landmark[18],
                                            hand_landmarks.landmark[19]),
                            calculate_angle(hand_landmarks.landmark[18], hand_landmarks.landmark[19],
                                            hand_landmarks.landmark[20]),
                            calculate_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[5],
                                            hand_landmarks.landmark[9]),
                            calculate_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[9],
                                            hand_landmarks.landmark[13]),
                            calculate_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[13],
                                            hand_landmarks.landmark[17])
                        ]
                        angle_buffer.append(angles)

                        # Log angle calculations
                        print(f"Calculated angles for {label}: {angles[:5]}...")  # Preview of first few angles

                        # Apply moving average to smooth the data
                        if len(landmark_buffer) == buffer_size:
                            smoothed_landmarks = moving_average(landmark_buffer, buffer_size)
                            smoothed_distances = moving_average(distance_buffer, buffer_size)
                            smoothed_angles = moving_average(angle_buffer, buffer_size)

                            feature_list.extend(smoothed_landmarks)
                            feature_list.extend(smoothed_distances)
                            feature_list.extend(smoothed_angles)

                            if args.dynamic:
                                sequence_buffer.append(feature_list)
                            else:
                                # Log the data that is about to be stored
                                print(f"Storing features for {label} in HDF5")  # Log before storing
                                log_queue.put((feature_list, label, args.dynamic))

                    log_queue.put((None, None, None))
                    log_thread.join()
                else:
                    print(f"No hand landmarks detected in image: {image_path}")  # Log if no hand was detected


if __name__ == "__main__":
    main()
