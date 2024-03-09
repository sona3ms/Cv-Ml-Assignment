# Font Matching with Computer Vision

This project aims to classify the fonts used in text images using computer vision techniques and deep learning. It provides recommendations for matching fonts with a confidence level.

## Project Structure

- `newfont.py`: Python script to generate training data by creating images with text written in different fonts.
- `custommodel.py`: Python script to define, train, and save a convolutional neural network (CNN) model for font classification.
- `customtrain.py`: Python script to load the trained model and perform font classification on input images.
- `requirements.txt`: List of Python packages required to run the project.

## Instructions
### Setting Up the Environment

1. Clone this repository to your local machine.
2. Install the required Python packages by running:
   pip install -r requirements.txt


### Data Creation

1. Run `newfont.py` to generate training data. This script creates images with the text "Hello, World!" written in different fonts.

### Model Training

1. Run `custommodel.py` to define, train, and save the CNN model for font classification. This script uses the generated training data to train the model.

### Model Evaluation

1. During model training, the performance of the model is evaluated on a separate validation set. You can monitor the training progress and evaluate the model's performance on unseen data.

### Font Classification

1. Run `customtrain.py` to load the trained model and perform font classification on input images. This script detects "Hello, World!" instances in input images and predicts the font used for each instance.

## Model Architecture

The CNN model architecture consists of convolutional layers followed by max-pooling layers to extract features from input images. These features are then flattened and passed through dense layers for classification. The final layer uses softmax activation to output probabilities for each font class.

## Usage

1. Ensure you have Python installed on your system.
2. Install the required packages by running `pip install -r requirements.txt`.
3. Fonts are avilable in 'font.txt'.
4. Place your font files (`.ttf` format) in the specified font directory.
5. Run the script `newfont.py`.
6. The generated images will be saved in the output directory specified.

## Requirements

- Python 3.x
- TensorFlow
- PIL (Python Imaging Library)
- pytesseract

