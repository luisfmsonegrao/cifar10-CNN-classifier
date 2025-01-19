
# Cifar10 CNN Classifier
This project defines a simple, custom Convolutional Neural Network for image classification and trains it on the **CIFAR-10** dataset.
It also provides tools to serve the trained model remotely (in *GCP*) in 3 different ways:  
    - Containerized Flask app  
    - Containerized Torchserve server  
    - Kubernetes containerized data-processing app + Torchserve server  


## Introduction
The SmallCNN submodule defines the **SmallCNN** class. This class defines a Convolutional Neural Network with 3 convolutional layers with batch normalization and average pooling between each pair of convolutional layers, followed by a sequence of fully connected layers with batch normalization. The network has 10 outputs that can be transformed into class probabilities for each of the 10 classes in the CIFAR10 dataset.

The *'CIFAR10dataset'* submodule defines a **CIFAR10dataset** class that wraps around the **CIFAR10** dataset class provided with torchvision.

The *'trained_model.py'* script inside the utils folder provides an example of how to train an instance of the **SmallCNN** on the CIFAR10 dataset.

## Deployment

This repository includes two docker images which can be used to serve a trained instance of the SmallCNN model as a *Flask* app or as a *torchserve* server, respectively.
The images can be built by running the command line scripts *'build_flask_app_image.cmd'* and *'build_torchserve_image.cmd'* from within the *'scripts'* folder.
Containers based on these images can be started in your local machine by running *'run_flask_app_container.cmd'* and *'run_torchserve_container.cmd'* stored in the *scripts* folder.
Alternatively, the images can be pushed to *GCP Artifact Registry* and run as remote containers using *GCP Cloud Run*. The *'scripts'* folder contains examples of how to push the images to the Artifact Registry. Please refer to the [GCP Artifact Registry](https://cloud.google.com/artifact-registry/docs) and [GCP Cloud Run](https://cloud.google.com/run/docs) documentation for more information.
Finally, the *'kubernetes-config'* folder defines deployments and services for a data processing app that prepocesses **CIFAR10** image data, and for a container with a model served with *torchserve*. The *'scripts'* folder contains *cmd* scripts to setup a *GKE* cluster, configure *kubectl* and start up services and deployments in the remote *GKE* cluster.

Note: currently the files in this repo are configured for deployment to *GKE*, so only the last deployment option described above will work out of the box.

## Inference
Once you have a running docker container, *'test_flask_app.py'* and *'test_torchserve_model.py'* from the *utils* folder can be used to perform inference on samples from the CIFAR10 dataset.
When you have setup a *kubernetes* cluster in *GKE* and all services and deployments are ready, *'test_gateway_app.py'* provides an ready-to-run example of how to performe inference on **CIFAR10** images.



