# Hand Gesture Recognition

This repository contains a hand gesture recognition system that uses both Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to classify static and dynamic hand gestures. The system is built using TensorFlow, OpenCV, and MediaPipe.

## Table of Contents

- Installation
- Usage
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- Training
- Models
- Contributing
- License

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/dvorakman/csci218-group-project.git
    cd csci218-group-project
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Application

To run the hand gesture recognition application, execute the following command:
```sh
python app.py
```

### Arguments

- [`--device`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fcardinal%2FDocuments%2FGitHub%2Fcsci218-group-project%2Fapp.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A39%2C%22character%22%3A25%7D%7D%5D%2C%221c4d8893-f6e8-4aff-84a4-0639fd0e15ac%22%5D "Go to definition"): The device index for the camera (default: 0).
- [`--width`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fcardinal%2FDocuments%2FGitHub%2Fcsci218-group-project%2Fapp.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A40%2C%22character%22%3A25%7D%7D%5D%2C%221c4d8893-f6e8-4aff-84a4-0639fd0e15ac%22%5D "Go to definition"): The width of the camera feed (default: 900).
- [`--height`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fcardinal%2FDocuments%2FGitHub%2Fcsci218-group-project%2Fapp.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A41%2C%22character%22%3A25%7D%7D%5D%2C%221c4d8893-f6e8-4aff-84a4-0639fd0e15ac%22%5D "Go to definition"): The height of the camera feed (default: 900).
- `--label`: The label for the data being collected (required).
- `--dynamic`: Flag to indicate dynamic gesture detection.

## Project Structure

```
.
├── .gitignore
├── app.py
├── data/
│   ├── dynamic/
│   │   ├── [clear].h5
│   │   ├── [submit].h5
│   │   ├── J.h5
│   │   ├── Test.h5
│   │   └── Z.h5
│   └── static/
│       ├── A.h5
│       ├── B.h5
│       ├── C.h5
│       ├── D.h5
│       ├── E.h5
│       ├── F.h5
│       ├── G.h5
│       ├── H.h5
│       ├── I.h5
│       ├── K.h5
│       ├── L.h5
│       ├── M.h5
│       ├── N.h5
│       ├── O.h5
│       ├── P.h5
│       ├── Q.h5
│       ├── R.h5
│       ├── S.h5
│       ├── T.h5
│       ├── U.h5
│       ├── V.h5
│       ├── W.h5
│       ├── X.h5
│       ├── Y.h5
├── LICENSE
├── models/
│   ├── dynamic_rnn.keras
│   └── static_cnn.keras
├── README.md
├── train_cnn.ipynb
├── train_rnn.ipynb
└── utils/
    ├── collect_data.py
    ├── cvfpscalc.py
    ├── dataset_information.py
    ├── max_camera_fps.py
    └── plot_dataset.py
```

## Data Collection

To collect data for training, use the `collect_data.py` script located in the `utils` directory. This script captures hand gestures and saves them in HDF5 format, and mark whether they are dynamic or not with the flag `--dynamic`

```sh
python utils/collect_data.py --label "your_label" --dynamic
```

## Training

### Training the CNN Model

To train the CNN model for static gesture recognition, use the `train_cnn.ipynb` notebook. This notebook loads the static gesture data and trains a CNN model.

### Training the RNN Model

To train the RNN model for dynamic gesture recognition, use the `train_rnn.ipynb` notebook. This notebook loads the dynamic gesture data and trains an RNN model.

## Models

The pre-trained models are stored in the `models` directory:
- `static_cnn.keras`: The CNN model for static gesture recognition.
- `dynamic_rnn.keras`: The RNN model for dynamic gesture recognition.
