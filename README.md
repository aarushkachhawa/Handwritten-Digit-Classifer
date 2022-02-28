[//]: # (Image References)

[image1]: ./images/predict_0.png "Prediction for digit 0"
[image2]: ./images/predict_1.png "Prediction for digit 1"
[image3]: ./images/predict_2.png "Prediction for digit 2"
[image4]: ./images/predict_3.png "Prediction for digit 3"
[image5]: ./images/predict_4.png "Prediction for digit 4"
[image6]: ./images/predict_5.png "Prediction for digit 5"
[image7]: ./images/predict_6.png "Prediction for digit 6"
[image8]: ./images/predict_7.png "Prediction for digit 7"
[image9]: ./images/predict_8.png "Prediction for digit 8"
[image10]: ./images/predict_9.png "Prediction for digit 9"
[image11]: ./images/blank_tool.png "Blank Tool"

## Project Overview

Welcome to the Handwritten Digit Recognition project! In this project, I will create a simple editor tool that will allow you to draw handwritten digits (0-9). For each digit you draw you can click a button, 'Predict the Digit'. The tool will then output a prediction of the digit and an accuracy score for the prediction. 

![Blank Tool][image11]

The tool supports 2 different implementations of neural networks to make the prediction:

* Convolutional Neural Network implemented with Keras 
* Multi Layer Perceptron using Pytorch. 

### Convolutional Neural Network

#### Data 
I started with downloading the data. I used the MNIST database of hand written digits to train and test the model. I loaded in the training and test data, split the training data into a training and validation set.

#### CNN architecture 
* 2D Convolutional layers, which can be thought of as stack of filtered images. 
* Maxpooling layers, which reduce the x-y size of an input, keeping only the most active pixels from the previous layer. 
* Linear + Dropout layers to avoid overfitting and produce a 10-dim output.

The more convolutional layers I include, the more complex patterns in color and shape a model can detect. Hence my model includes 2 convolutional layers as well as linear layers + dropout in between to avoid overfitting.

#### Loss optimization 
I used a loss and optimization function that is best suited for this classification task namely keras categorical_crossentropy loss funcion and Adadelta optimizer

### Multi Layer Perceptron

#### Data Loading
I used the MNIST database of handwritten digits to train the MLP. I set a batch_size to load more data at a time. I created DataLoaders for training and test datasets

#### Network Architecture 
The architecture will be responsible for seeing as input a 784-dim Tensor of pixel values for each image, and producing a Tensor of length 10 (my number of classes) that indicates the class scores for an input image. I used two hidden layers and dropout to avoid overfitting.

#### Loss Function and Optimizer 
I used cross-entropy loss for classification and stochastic gradient descent optimizer  

#### Training the Network 
The steps for training/learning from a batch of data are described in the comments below:

* Clear the gradients of all optimized variables
* Forward pass: compute predicted outputs by passing inputs to the model
* Calculate the loss
* Backward pass: compute gradient of the loss with respect to model parameters
* Perform a single optimization step (parameter update)
* Update average training loss

### Demo of predictions

Here are some sample outputs of the tool:

![Prediction for digit 0][image1]
![Prediction for digit 1][image2]
![Prediction for digit 2][image3]
![Prediction for digit 3][image4]
![Prediction for digit 4][image5]
![Prediction for digit 5][image6]
![Prediction for digit 6][image7]
![Prediction for digit 7][image8]
![Prediction for digit 8][image9]
![Prediction for digit 9][image10]




