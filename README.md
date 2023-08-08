# Convolutional_Neural_Network


Convolutional Neural Network (CNN) for Image Classification

This repository contains code for training a simple CNN model for image classification using PyTorch. The trained model can be saved and later used to make predictions on new images.


1. Clone this repository to your local machine:


```sh
   git clone https://github.com/Jibrilmamo/convolutional-neural-network.git
   cd convolutional-neural-network
   ```

2. Install the required dependencies:

   ```sh
   pip install torch torchvision
   ```

3. Prepare Your Data:

   - Place your training data (numpy arrays or images) in the appropriate directory.
   - Update the data loading code in the script accordingly.

4. Train the Model:

   Run the following command to train the CNN model:

   ```sh
   python convolutional_neural_network.py train
   ```

   The trained model's state will be saved as `trained_model.pth`.


## Script

The `train_cnn.py` script handles training.
The `cnn.py` the Neural Network.
the `convolution_dem` a demo of how the post convolution and max pooling results look.
the `trained_model.pth` a trained model


## Customization

You can customize the model architecture, hyperparameters, and other settings in the `CNN` class in the script.

## Acknowledgments

This project was created as a learning exercise and is based on PyTorch tutorials and official documentation.
