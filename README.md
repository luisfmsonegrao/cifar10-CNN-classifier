# Cifar10 CNN Classifier
This project defines a simple, custom Convolutional Neural Network for image classification and trains it on the CIFAR-10 dataset.
Included in the project are docker images to serve the model in a remote container as a flask app or as a torchserve server.

## Introduction
The SmallCNN submodule defines the **SmallCNN** class. This class defines a Convolutional Neural Network with 3 convolutional layers with batch normalization and average pooling between each pair of convolutional layers, followed by a sequence of fully collected layers with batch normalization. The network has 10 outputs that can be transformed into probabilities for each of the CIFAR-10 dataset classes.

The CIFAR10dataset submodule defines a **CIFAR10dataset** class that wraps around the **CIFAR10** dataset class provided with torchvision.

The *trained_model.py* script inside the utils folder provides an example of how to train an instance of the **SmallCNN** on the CIFAR10 dataset.

## Deployment

This repository provides two docker images that can be used to serve a trained instance of the SmallCNN model as a Flask app and as a torchserve server, respectively.
The images can be built by running the command line scripts *'build_flask_app_image.cmd'* and *'build_torchserve_image.cmd'* from within the *scripts* folder.
Containers based on these images can be started in your local machine by running *'run_flask_app_container.cmd'* and *'run_torchserve_container.cmd'* stored in the *scripts* folder.
Alternatively, the images can be pushed to GCP Artifact Registry and run as remote containers. The *scripts* folder contains examples of how to push the images to the Artifact Registry. Please refer to the 
 [GCP Artifact Registry](https://cloud.google.com/artifact-registry/docs) documentation for more information

## Inference
Once you have a running docker container, *'test_flask_app.py'* and *'test_torchserve_model.py'* can be used to perform inference on samples from the CIFAR10 dataset.



