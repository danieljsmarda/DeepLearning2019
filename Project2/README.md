# DeepLearning2019
Project 2 for Masters Course EE-559 taught at EPFL (Lausanne, Switzerland) by Professor Francois Fleuret.

## Overview
The purpose of this project was to build a mini deep-learning framework based on PyTorch. The framework developed here uses PyTorch's basic Tensor objects but explicitly implements the abstractions necessary for building models, such as those in torch.nn.

## Code Structure
Detailed descriptions of each of the functions can be found in the file report.pdf. Generally, there are two files:
* `test.py`
* `helpers.py`

`test.py` simply calls the functions implemented in `helpers.py`. `helpers.py` is split into several sections:
* Data creation/handling: these fuctions generate the sample data and split the data into training, validation, and test sets.
* Modules: Contains the network modules. A Sequential object represnts a linear network, and the `forward`, `backward`, and `param` methods of the Sequential object essentially call the corresponding methods of the modules that comprise the Sequential object. The `Sequential` object can be composed
of `Linear` layers, `Tanh` layers and `ReLu` layers. This section also contains a `SGD` module which controls how the model takes optimization steps during training.
* Loss Functions: Implements MSE and the corresponding derivative for the backpropogation as global functions.
* Training and Testing: The more complicated function, `train_data`, trains the network based on input data, learning rate, targets, and a `Sequential` model.
It keeps track of how much time the model took to train and prints loss values at intervals at epochs so that the user can view the training rate as the model trains. 

## Running the tests

The `test.py` file should be run with no arguments, for example:

```python test.py```

The defaults for the training hyperparameters can be found and modified in lines 25-29 of `test.py` to decrease training time. 

For each of the models, the test.py file trains the model and prints the loss at the first epoch, the lost at the last epoch, and the overall test accuracy.

## Built With

* [PyTorch](https://pytorch.org/) - Building and training the neural networks


## Authors

* [Youssef Janjar](https://github.com/YoussefJanjar)
* [Daniel Smarda](https://github.com/danieljsmarda/)

## Acknowledgments

Many thanks to Professor Fran&#231;ois Fleuret for his [teaching materials (slides and practicals)](https://fleuret.org/ee559/index.html) and the accompanying team of teaching assistants. 
