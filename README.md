# neural-network
HKOI Training Team 2017 Deep Neural Network (Tensorflow)

Slides: http://assets.hkoi.org/training2017/nn.pdf

# Notes

The `data` folder contains the preprocessed MNIST data set.

The `tfpreprocess` folder mentioned in the slides is placed [here](preprocess). It contains the Art Class data set. Art Class data set is licensed under the [CC-BY](https://creativecommons.org/licenses/by/3.0/) from the IOI 2013 organizer.

## Create your own dataset

Create a folder inside `preprocess`. Inside the folder there should be a file called `classes.txt`. Put the class names there in separate lines. Then the folder should contain two folders: `training` and `validation`. Inside each of the two folders there should be folders named after the classes. You can place the PNGs and JPGS there (no filename restrictions).
