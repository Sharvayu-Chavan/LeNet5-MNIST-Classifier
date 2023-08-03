# LeNet5-MNIST-Classifier
A simple convolutional neural network using LeNet architecture to classify numbers from the MNIST dataset.

## MNIST Digit Classification using LeNet Architecture

This repository contains code for training a LeNet architecture model on the MNIST dataset using PyTorch. The trained model can classify handwritten digits into 10 different categories.

## Dataset
The code automatically downloads the MNIST dataset during execution. It consists of 60,000 training and 10,000 test images, each with a size of 28x28 pixels in grayscale. The dataset is split into training and test sets, which are used to train and evaluate the model.

## Model Architecture
The LeNet model architecture consists of convolutional and fully connected layers. It takes as input a grayscale image and processes it through convolutional and pooling layers, followed by fully connected layers for classification. The architecture is designed for efficient digit recognition.

## Prerequisites
- Python 3
- PyTorch
- Torchvision

## Installation
Clone this repository to your local machine: 
```shell
git clone https://github.com/Sharvayu-Chavan/LeNet5-MNIST-Classifier.git
```

## Usage
1. Install the required dependencies using pip:
```shell
pip install torch torchvision
```
3. Run the `LeNetMNIST_CNN.py` script in your environment to train the LeNet model:
```shell
python LeNetMNIST_CNN.py
```
4. The script will automatically download the dataset, train the LeNet model, and display the training progress and test accuracy.

5. After training, the script will save the trained model as `model.pth`.

## License
MIT License

Feel free to use and modify the code according to the license terms.

## Acknowledgments
The MNIST dataset was originally created by Yann LeCun and Corinna Cortes.

The PyTorch and [MNIST classification using LeNet on Pytorch](https://www.kaggle.com/code/yogeshrampariya/mnist-classification-using-lenet-on-pytorch) tutorials and examples inspire the code in this repository.


