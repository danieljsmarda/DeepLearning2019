# DeepLearning2019
Project 1 for Masters Course EE-559 taught at EPFL (Lausanne, Switzerland) by Professor Francois Fleuret.

## Overview
The purpose of this project was to investigate the effect of weight sharing and auxiliary losses on a deep neural network. Given two 14x14 MNIST digit arrays, the goal was to classify whether the digit in the first channel had a value greater than or less than the value in the second channel. We started with a simple convolutional network with two rounds of convolution and max-pooling followed by a fully-connected linear layer. We then modified this basic structure to implement weight sharing, where the two channels were split at the beginning, instructed to share weights, then merged prior to the final layer. Finally, this weight-sharing model was modified to also incorporate an auxiliary loss of whether the network correctly classified each channel's 0-9 MNIST value. Further details and diagrams can be found in report.pdf.

## Code Structure
There are 5 main code files:
* SimpleCNN.py
* AuxModel.py
* WSharingModel.py
* test.py
* helpers.py

The first 3 files are known as the model files. Each of the model files contain 2 main parts:
* Base module class (SimpleModel/WSModel/AuxModel): This defines the structure of the network.
* Training functions (train_model/train_model_WS/train_model_AM): These functions define how the networks defined in the base module classes are trained.

In addition, both the weight sharing model file and the auxiliary loss model file contain exploration classes (WSModel1, AuxModel1). These classes are very similar to the base module classes. The base module classes contain the hyperparameters with which we achieved best performance; WSModel1 and AuxModel1 serve as classes that can be modified, retested, and compared to the base module classes.


## Running the tests

The `test.py` file should be run with no arguments, for example:

```python test.py```

The defaults for the training hyperparameters are 50 epochs and a learning rate of 0.1. With an Intil i7 this file takes about 2 minutes to run; to speed up the training time, reduce the NB_EPOCHS variable in line 39 of helpers.py.


### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc


