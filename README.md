# Neural-Network
Implementing a neural network from scratch to identify hand written letters

The dataset used here for this program was a subset of OCR (Optical Character Recognition) dataset.

A single hidden layer neural network has been built from scratch.
The acitvation functions are sigmoid and there is a softmax output layer
2 different initializations have been used:
1. Zero for all weights
2. Random initialization between -0.1 to 0.1

SGD (Stochastic Gradient Descent) has been used to find the values of the parameters for the neural network.

The different inputs/outputs are (using sys.argv[n]):
1. train input file
2. validation input file
3. train output file
4. validation output file
5. metrics output (train and validation error)
6. number of epochs
7. number of hidden units in the hidden layer
8. initialization method (1 or 2)
9. learning rate
