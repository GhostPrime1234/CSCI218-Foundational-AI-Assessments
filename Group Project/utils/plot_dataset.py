import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import h5py
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic", action='store_true', help='Dynamic gesture dataset')
    parser.add_argument("--label", type=str, help='Filter by gesture label')
    args = parser.parse_args()
    args.label = args.label.upper() if args.label else None
    return args

def read_h5_files(directory, label=None):
    X = []
    y = []
    for file in os.listdir(directory):
        if file.endswith(".h5"):
            if label and file.split(".")[0] != label:
                continue
            filepath = os.path.join(directory, file)
            with h5py.File(filepath, "r") as f:
                for dataset_name in f.keys():
                    # Read the dataset (each sequence has variable length)
                    data = f[dataset_name][:]
                    file_label = file.split(".")[0]  # Extract label from filename
                    
                    # Append data and label
                    X.append(data)
                    y.append(file_label)
    return X, np.array(y)

def update_dynamic(frame, data, label):
    
    ax1.cla()  # Clear the previous frame
    ax2.cla()
    ax3.cla()
    
    # Set background color to black
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    ax3.set_facecolor('black')
    
    # # Calculate the start and end indices for the current frame
    # start_idx = frame * feature_list_length
    # end_idx = start_idx + num_landmarks
    
    # # Extract x, y, z coordinates for landmarks
    # landmark_frame = data.iloc[0, start_idx:end_idx].values
    landmark_frame = data[frame][0][0, :num_landmarks]
    
    x = landmark_frame[0::3]
    y = landmark_frame[1::3]
    z = landmark_frame[2::3]
    
    # Plot the landmarks
    ax1.scatter(x, y, z, c='white', marker='o')
    ax1.set_xlabel('X', color='white')
    ax1.set_ylabel('Y', color='white')
    ax1.set_zlabel('Z', color='white')
    ax1.set_title(f'Dynamic Landmarks - Label {label[frame]} - Frame {frame}', color='white')
    ax1.tick_params(colors='white')
    
    # Draw lines between the landmarks
    for connection in connections:
        start, end = connection
        ax1.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 'white')
    
    # Extract and plot the distances
    distance_frame = data[frame][0][0, num_landmarks:num_landmarks + 12]
    ax2.plot(distance_frame, 'wo-')
    ax2.set_xlabel('Distance Index', color='white')
    ax2.set_ylabel('Distance', color='white')
    ax2.set_title(f'Dynamic Distances - Frame {frame}', color='white')
    ax2.tick_params(colors='white')
    
    # Extract and plot the angles
    angle_frame = data[frame][0][0, num_landmarks+12:num_landmarks + 25]
    ax3.plot(angle_frame, 'wo-')
    ax3.set_xlabel('Angle Index', color='white')
    ax3.set_ylabel('Angle (degrees)', color='white')
    ax3.set_title(f'Dynamic Angles - Frame {frame}', color='white')
    ax3.tick_params(colors='white')
    
    # Append the current positions of the fingertips to their respective lists
    thumb_positions.append([x[4], y[4], z[4]])
    index_positions.append([x[8], y[8], z[8]])
    middle_positions.append([x[12], y[12], z[12]])
    ring_positions.append([x[16], y[16], z[16]])
    pinky_positions.append([x[20], y[20], z[20]])
    
    # Keep only the last 'buffer_size' positions in the lists
    if len(thumb_positions) > buffer_size:
        thumb_positions.pop(0)
    if len(index_positions) > buffer_size:
        index_positions.pop(0)
    if len(middle_positions) > buffer_size:
        middle_positions.pop(0)
    if len(ring_positions) > buffer_size:
        ring_positions.pop(0)
    if len(pinky_positions) > buffer_size:
        pinky_positions.pop(0)
    
    # Plot the paths taken by the fingertips with fading effect
    for i in range(1, len(thumb_positions)):
        alpha = i / len(thumb_positions)
        ax1.plot([thumb_positions[i-1][0], thumb_positions[i][0]], 
                 [thumb_positions[i-1][1], thumb_positions[i][1]], 
                 [thumb_positions[i-1][2], thumb_positions[i][2]], c='yellow', alpha=alpha)
    for i in range(1, len(index_positions)):
        alpha = i / len(index_positions)
        ax1.plot([index_positions[i-1][0], index_positions[i][0]], 
                 [index_positions[i-1][1], index_positions[i][1]], 
                 [index_positions[i-1][2], index_positions[i][2]], c='cyan', alpha=alpha)
    for i in range(1, len(middle_positions)):
        alpha = i / len(middle_positions)
        ax1.plot([middle_positions[i-1][0], middle_positions[i][0]], 
                 [middle_positions[i-1][1], middle_positions[i][1]], 
                 [middle_positions[i-1][2], middle_positions[i][2]], c='magenta', alpha=alpha)
    for i in range(1, len(ring_positions)):
        alpha = i / len(ring_positions)
        ax1.plot([ring_positions[i-1][0], ring_positions[i][0]], 
                 [ring_positions[i-1][1], ring_positions[i][1]], 
                 [ring_positions[i-1][2], ring_positions[i][2]], c='green', alpha=alpha)
    for i in range(1, len(pinky_positions)):
        alpha = i / len(pinky_positions)
        ax1.plot([pinky_positions[i-1][0], pinky_positions[i][0]], 
                 [pinky_positions[i-1][1], pinky_positions[i][1]], 
                 [pinky_positions[i-1][2], pinky_positions[i][2]], c='blue', alpha=alpha)

def update_static(frame, data, label):
    ax1.cla()  # Clear the previous frame
    ax2.cla()
    ax3.cla()
    
    # Set background color to black
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    ax3.set_facecolor('black')

    landmark_frame = data[frame][0, :num_landmarks]

    x = landmark_frame[0::3]
    y = landmark_frame[1::3]
    z = landmark_frame[2::3]

    # Plot the landmarks
    ax1.scatter(x, y, z, c='white', marker='o')
    ax1.set_xlabel('X', color='white')
    ax1.set_ylabel('Y', color='white')
    ax1.set_zlabel('Z', color='white')
    ax1.set_title(f'Static Landmarks - Frame {frame} - Label {label[frame]}', color='white')
    ax1.tick_params(colors='white')
    
    # Draw lines between the landmarks
    for connection in connections:
        start, end = connection
        ax1.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 'white')
    
    # Plot the distances
    distance_frame = data[frame][0, num_landmarks:num_landmarks + 12]
    ax2.plot(distance_frame, 'wo-')
    ax2.set_xlabel('Distance Index', color='white')
    ax2.set_ylabel('Distance', color='white')
    ax2.set_title('Static Distances', color='white')
    ax2.tick_params(colors='white')
    
    # Plot the angles
    angle_frame = data[frame][0, num_landmarks+12:num_landmarks + 25]
    ax3.plot(angle_frame, 'wo-')
    ax3.set_xlabel('Angle Index', color='white')
    ax3.set_ylabel('Angle (degrees)', color='white')
    ax3.set_title('Static Angles', color='white')
    ax3.tick_params(colors='white')

def main():
    args = get_args()

    if args.dynamic:
        data_dir = 'data/dynamic'
    else:
        data_dir = 'data/static'

    X, y = read_h5_files(data_dir, args.label)

    global connections, num_landmarks

    # Define connections between landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]

    # Number of landmarks (21 points with x, y, z coordinates)
    num_landmarks = 21 * 3

    if args.dynamic:
        global buffer_size, feature_list_length, thumb_positions, index_positions, middle_positions, ring_positions, pinky_positions

        buffer_size = 10  # Number of frames to keep in the buffer
        feature_list_length = 88  # Number of features in each frame

        thumb_positions = []
        index_positions = []
        middle_positions = []
        ring_positions = []
        pinky_positions = []

    global fig, ax1, ax2, ax3
    # Create the figure and subplots
    fig = plt.figure(figsize=(39, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Set figure background color to black
    fig.patch.set_facecolor('black')

    if args.dynamic:
        ani = FuncAnimation(fig, update_dynamic, frames=len(X), fargs=(X, y,), repeat=True, interval=250)
    else:
        ani = FuncAnimation(fig, update_static, frames=len(X), fargs=(X, y,), repeat=True, interval=250)

    plt.show()

if __name__ == '__main__':
    main()