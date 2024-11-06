import os
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import argparse
import math
from collections import deque
import pandas as pd  # Import pandas for CSV handling


# Load the pre-trained models
# cnn_model = tf.keras.models.load_model('models/static_cnn.keras')
# rnn_model = tf.keras.models.load_model('models/dynamic_rnn.keras')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images.",
                        default="data/static/asl_dataset")
    args = parser.parse_args()
    return args


def calculate_distance(point1, point2):
    distance = math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)
    return distance


def calculate_angle(point1, point2, point3):
    a = np.array([point1.x, point1.y, point1.z])
    b = np.array([point2.x, point2.y, point2.z])
    c = np.array([point3.x, point3.y, point3.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)
    return angle_degrees


def moving_average(data, window_size):
    average = np.mean(list(data)[-window_size:], axis=0)
    return average


def draw_info(image, gesture, model_used, confidence, hand_label):
    text_pos_y = {"Left": 30, "Right": 200}

    cv2.putText(image, f"Gesture ({hand_label}): {gesture}",
                (10, text_pos_y[hand_label]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Model: {model_used}", (10, text_pos_y[hand_label] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Confidence: {confidence:.2f}",
                (10, text_pos_y[hand_label] + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    return image


def draw_hand_landmarks(image, hand_landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing.draw_landmarks(
        image, hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    return image


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return image


def process_folder(folder_path):
    image_data = []
    # Traverse all subdirectories and collect image paths with their labels
    for root, _, files in os.walk(folder_path):
        folder_name = os.path.basename(root)  # Use the folder name as the label
        for file in files:
            if file.endswith(('.jpeg', '.jpg', '.png')):
                image_path = os.path.join(root, file)
                image_data.append((image_path, folder_name))  # Store as tuple (image_path, label)
    return image_data


def calculate_hand_distances(hand_landmarks):
    """Calculate distances for hand landmarks."""
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
    return distances


def calculate_hand_angles(hand_landmarks):
    """Calculate angles for hand landmarks."""
    angles = [
        calculate_angle(hand_landmarks.landmark[1], hand_landmarks.landmark[2], hand_landmarks.landmark[3]),
        calculate_angle(hand_landmarks.landmark[2], hand_landmarks.landmark[3], hand_landmarks.landmark[4]),
        calculate_angle(hand_landmarks.landmark[5], hand_landmarks.landmark[6], hand_landmarks.landmark[7]),
        calculate_angle(hand_landmarks.landmark[6], hand_landmarks.landmark[7], hand_landmarks.landmark[8]),
        calculate_angle(hand_landmarks.landmark[9], hand_landmarks.landmark[10], hand_landmarks.landmark[11]),
        calculate_angle(hand_landmarks.landmark[10], hand_landmarks.landmark[11], hand_landmarks.landmark[12]),
        calculate_angle(hand_landmarks.landmark[13], hand_landmarks.landmark[14], hand_landmarks.landmark[15]),
        calculate_angle(hand_landmarks.landmark[14], hand_landmarks.landmark[15], hand_landmarks.landmark[16]),
        calculate_angle(hand_landmarks.landmark[17], hand_landmarks.landmark[18], hand_landmarks.landmark[19]),
        calculate_angle(hand_landmarks.landmark[18], hand_landmarks.landmark[19], hand_landmarks.landmark[20]),
        calculate_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[5], hand_landmarks.landmark[9]),
        calculate_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[9], hand_landmarks.landmark[13]),
        calculate_angle(hand_landmarks.landmark[0], hand_landmarks.landmark[13], hand_landmarks.landmark[17])
    ]
    return angles


def main():
    args = get_args()
    image_data = process_folder(args.folder_path)
    if not image_data:
        return

    mp_hands = mp.solutions.hands

    buffer_size = 5
    sequence_length = 10

    # Buffers for each hand
    buffers = {
        "Left": {"landmark": deque(maxlen=buffer_size),
                 "distance": deque(maxlen=buffer_size),
                 "angle": deque(maxlen=buffer_size),
                 "sequence": deque(maxlen=sequence_length)},
        "Right": {"landmark": deque(maxlen=buffer_size),
                  "distance": deque(maxlen=buffer_size),
                  "angle": deque(maxlen=buffer_size),
                  "sequence": deque(maxlen=sequence_length)}
    }

    # Dictionary to hold data for each gesture (A-Z, 0-9)
    gesture_data = {gesture: [] for gesture in os.listdir(args.folder_path) if os.path.isdir(os.path.join(args.folder_path, gesture))}

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                        model_complexity=1, min_detection_confidence=0.75) as hands:

        for image_path, label in image_data:
            image = process_image(image_path)
            if image is None:
                continue

            # Process the image
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for i, (hand_landmarks, hand_handedness) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)):
                    hand_label = hand_handedness.classification[0].label  # 'Left' or 'Right'

                    image = draw_hand_landmarks(image, hand_landmarks)

                    # Store the landmarks
                    buffers[hand_label]["landmark"].append(hand_landmarks)

                    # Calculate and store distances
                    distances = calculate_hand_distances(hand_landmarks)
                    buffers[hand_label]["distance"].append(distances)

                    # Calculate and store angles
                    angles = calculate_hand_angles(hand_landmarks)
                    buffers[hand_label]["angle"].append(angles)

                    # Example of gesture recognition logic (replace with your model's prediction)
                    gesture = label  # Assuming the folder name is the gesture
                    gesture_data[gesture].append([distances, angles])

                    # Draw gesture information on the image
                    confidence = 1.0  # Placeholder for model confidence
                    image = draw_info(image, gesture, "Static Model", confidence, hand_label)

            cv2.imshow("Image", image)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Convert gesture data to DataFrame and save to CSV
    for gesture, data in gesture_data.items():
        df = pd.DataFrame(data, columns=["Distances", "Angles"])
        df.to_csv(f"gesture_data_{gesture}.csv", index=False)


if __name__ == "__main__":
    main()
