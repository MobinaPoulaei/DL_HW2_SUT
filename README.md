# Deep Learning Homework 2 - Sharif University of Technology
### Instructor: Dr. Mahdieh Soleymani

This repository contains my solutions and code for the second homework assignment of the Deep Learning course at Sharif University of Technology. This assignment focuses on more advanced topics in convolutional neural networks, including batch normalization, specialized architectures, and computer vision tasks like classification, segmentation, and object detection.

---

### **Theoretical Section**

The theoretical portion of this homework covers advanced concepts in deep learning architectures and their mathematical foundations. My answers are submitted in a single PDF file as required by the course. The topics covered are:

* **Batch Normalization**: Comparing its application in fully connected vs. convolutional networks and its behavior during training and inference.
* **Dilated Convolution**: Analysis of receptive fields and computational costs.
* **ROI Alignment**: Understanding its mechanism and application in object detection.
* **Convolution Gradient**: Deriving gradients for a 1D convolution operation.
* **Gradient Vanishing**: A theoretical analysis of the gradient vanishing problem and how skip connections help mitigate it.
* **MobileNet**: An in-depth look at lightweight architectures, including depthwise separable convolutions, inverted residual blocks, and neural architecture search (NAS).

### **Practical Section**

This part of the homework involves implementing and analyzing different convolutional neural network architectures for computer vision tasks. The code is organized into separate notebooks for each question.

* **Classification**: In this notebook, we perform **classification** on images from the **CIFAR10** dataset using CNNs. We load the data, apply necessary transformations, and design and train a convolutional network. The notebook also includes a deep analysis of the feature space using KNN, clustering, and visualization of intermediate layer outputs. In the second part, we explore **transfer learning** by adapting the trained model to the **CIFAR100** dataset, retraining the final layer, and evaluating its generalization ability.

* **Segmentation**: This notebook is dedicated to building a **U-Net** for semantic image segmentation on the **CARLA** self-driving car dataset. The goal is to predict a precise label for every pixel in an image.  The assignment explains the difference between a regular CNN and a U-Net, and demonstrates how to apply sparse categorical cross-entropy for pixel-wise prediction.

* **Object Detection**: In this assignment, we build a two-stage license plate recognition system. The first stage focuses on **license plate detection (LPD)** to locate the plates in an image. The second stage, **license plate recognition (LPR)**, involves recognizing individual characters within the detected plates. This approach is an alternative to general OCR, leveraging the fixed and predictable structure of license plates.
