import tensorflow as tf

# Configure TensorFlow to use GPU if available
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

# Rest of your imports
import cv2
import mediapipe as mp
import numpy as np
import argparse
import h5py
import math
from collections import deque
import os
import time

# # Load the pre-trained models
# with tf.device('/GPU:0' if gpus else '/CPU:0'):
cnn_model = tf.keras.models.load_model('models/static_cnn.keras')
rnn_model = tf.keras.models.load_model('models/dynamic_rnn.keras')

# Define the hand gesture labels
# static_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']]
static_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
dynamic_labels = ['J', 'Z']
all_labels = static_labels + dynamic_labels

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    return args

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

def draw_info(image, fps, gesture, model_used, confidence):
    cv2.putText(
        image,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Gesture: {gesture}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Model: {model_used}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Confidence: {confidence:.2f}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return image

def draw_hand_landmarks(image, hand_landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    return image

def main():
    args = get_args()

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands

    # Buffer to store previous frames for smoothing
    buffer_size = 5
    landmark_buffer = deque(maxlen=buffer_size)
    distance_buffer = deque(maxlen=buffer_size)
    angle_buffer = deque(maxlen=buffer_size)

    # Buffer to store sequences for dynamic gesture detection
    sequence_length = 10 # Sliding window size
    sequence_buffer = deque(maxlen=sequence_length)

    # FPS calculation
    prev_frame_time = 0
    current_gesture = "None"
    model_used = "None"
    confidence = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75) as hands:
        
        # with tf.device('/GPU:0' if gpus else '/CPU:0'):
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Calculate FPS
            current_frame_time = time.time()
            fps = 1 / (current_frame_time - prev_frame_time)
            prev_frame_time = current_frame_time

            image = cv2.flip(image, 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
                for hand_landmarks, hand_world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                    # Draw hand landmarks
                    image = draw_hand_landmarks(image, hand_landmarks)

                    feature_list = []
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    landmark_buffer.append(landmarks)
                    
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

                    # Calculate angles between joints
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
                    angle_buffer.append(angles)

                    # Apply moving average to smooth the data
                    if len(landmark_buffer) == buffer_size:
                        smoothed_landmarks = moving_average(landmark_buffer, buffer_size)
                        smoothed_distances = moving_average(distance_buffer, buffer_size)
                        smoothed_angles = moving_average(angle_buffer, buffer_size)
                        
                        feature_list.extend(smoothed_landmarks)
                        feature_list.extend(smoothed_distances)
                        feature_list.extend(smoothed_angles)

                        sequence_buffer.append(feature_list)

                        # Predict using both models
                        if len(sequence_buffer) == sequence_length:
                            cnn_input = np.array(feature_list).reshape(1, 1, 88)
                            rnn_input = np.array(sequence_buffer).reshape(1, sequence_length, 88)

                            cnn_prediction = cnn_model.predict(cnn_input, verbose=0)
                            rnn_prediction = rnn_model.predict(rnn_input, verbose=0)

                            # Decision mechanism
                            CNN_THRESHOLD = 0.60  # Increased to be more selective
                            RNN_THRESHOLD = 0.99 # Slightly increased for dynamic gestures

                            cnn_confidence = np.max(cnn_prediction)
                            rnn_confidence = np.max(rnn_prediction)
                            # rnn_confidence = 0
                            if rnn_confidence > RNN_THRESHOLD:
                                # Only trust RNN if it's extremely confident
                                current_gesture = dynamic_labels[np.argmax(rnn_prediction)]
                                model_used = "RNN"
                                confidence = rnn_confidence
                            elif cnn_confidence > CNN_THRESHOLD:
                                # If CNN is very confident, trust it for static gestures
                                current_gesture = static_labels[np.argmax(cnn_prediction)]
                                model_used = "CNN"
                                confidence = rnn_confidence
                            else:
                                # neither model is very confident, compare their confidence
                                if rnn_confidence > cnn_confidence * 0.95:
                                    current_gesture = dynamic_labels[np.argmax(rnn_prediction)]
                                    model_used = "RNN"
                                    confidence = rnn_confidence
                                else:  # Give slight advantage to CNN
                                    current_gesture = static_labels[np.argmax(cnn_prediction)]
                                    model_used = "CNN"
                                    confidence = cnn_confidence

                            print(f"""
                                    RNN: {dynamic_labels[np.argmax(rnn_prediction)]} (conf: {rnn_confidence:.4f})
                                    CNN: {static_labels[np.argmax(cnn_prediction)]} (conf: {cnn_confidence:.4f})"
                                    Chosen: {model_used} - {current_gesture} (conf: {confidence:.4f})
                                    """, end="\r")

            # Draw info on the image
            image = draw_info(image, fps, current_gesture, model_used, confidence)

            cv2.imshow('Hand Gesture Recognition', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()