# Facial Emotion Recognition

This repository contains a facial emotion recognition project implemented using Convolutional Neural Networks (CNNs). The project is written in Python using the Keras library for building and training the deep learning model.

## Getting Started

To use this project, follow the steps below:

1. Clone the repository:

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

2. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn keras opencv-python
```

3. Download the dataset `fer2013.csv` and place it in the root directory of the repository.

## Dataset Preprocessing

The first step in the project is to preprocess the dataset. The script `facerec.py` reads the `fer2013.csv` file and extracts the facial expression data. It then converts the pixel data into numpy arrays and stores them for training the model.

To preprocess the dataset, run:

```bash
python facerec.py
```

This script will perform the following steps:
- Read the `fer2013.csv` file.
- Extract the facial expression data and labels.
- Normalize and save the features in `fdataX.npy`.
- Save the labels in one-hot encoded format in `flabels.npy`.

## Training the Model

After preprocessing the dataset, the CNN model is designed and trained using the processed data. The model architecture consists of multiple convolutional and pooling layers followed by dense layers for classification.

To train the model, run:

```
python facerec.py
```

This script will:
- Load the preprocessed data from `fdataX.npy` and `flabels.npy`.
- Split the data into training, validation, and testing sets.
- Design and compile the CNN model using the Adam optimizer and categorical crossentropy loss.
- Train the model using the training data and validate it on the validation set.
- Save the trained model architecture in `s.json` and model weights in `facemodel.h5`.

## Facial Emotion Recognition

Once the model is trained, you can use it to recognize facial emotions from an input image. The script `facetest.py` loads the trained model and applies it to the input image.

To recognize emotions from an image, run:

```
python facetest.py
```

This script will:
- Load the trained model from `s.json` and `facemodel.h5`.
- Load the input image (replace `"angry.jpg"` with the path to your image).
- Detect faces in the image using Haar cascades.
- Crop and preprocess the detected face.
- Perform emotion prediction using the trained model.
- Display the image with the predicted emotion label.

## Conclusion

This project demonstrates facial emotion recognition using deep learning techniques. Feel free to modify the model architecture, train on different datasets, or use it in your own applications. For any questions or issues, please open an issue in the repository.
