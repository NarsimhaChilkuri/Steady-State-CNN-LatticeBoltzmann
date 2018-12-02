# Data-driven Steady State Flow Prediction

Use of machine learning tools to solve problems in physics (condensed matter, fluid mechanics) and chemistry (reaction prediction, drug discovery) is picking up. As far as I am aware, two professors in Canada, [Roger Melko](http://www.science.uwaterloo.ca/~rgmelko/index.html) of UWarteloo and [Alan Aspuru](http://matter.toronto.edu/machine-learning/) of UToronto, are at the forefront of this type of research.  In this, I'll be looking at an application of machine learning in the field of Fluid Mechanics by reimplementing, as far as I can, a paper titled [Convolutional Neural Networks for Steady Flow Approximation](https://autodeskresearch.com/publications/convolutional-neural-networks-steady-flow-approximation).

<img src="Images/blobs.jpg" width="300" height="400">

Generating the dataset using traditional CFD methods is the hardest part about the implementiation. Since the authors of the paper have not made the dataset available to the public, I had to do some things differently. In the paper, they use a dataset of containng 100,000 images; the input and output of the network are images of size 256x128 and 256x128x2 respectively. All I could manage, thanks to Oliver Henning, is a dataset of 3000 training images. Because of this constraint, I decided to modify the network so that it uses far fewer parameters; Insted of a netword that predicts x and y components of velocity field in the form of 256x128x2, I decided to go with a network that predicts just the magnitude of the velocity field in the form of 256x128.

![cnn_result](Images/__results___3_1.png)

![cnn_result](Images/__results___4_1.png)
