# Plant Segmentation from Aerial Field Images using Attention-Based U-Net Model

## Table of contents
- [Abstract](#abstract)
- [U-Net](#u-net)
- [Attention Mechanism](#attention-mechanism)
- [Attention U-Net](#attention-u-net)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Training Parameters](#training-parameters)

## Abstract
The Attention U-Net model is a convolutional neural network that incorporates attention mechanisms to selectively focus on the most relevant regions of the image during training. We perform a performance analysis of the Attention U-Net model in comparison to the standard U-Net architecture and evaluate the segmentation results using standard metrics. Our results show that the Attention U-Net model outperforms the U-Net model in terms of segmentation accuracy, particularly when working with limited sample-sized datasets. The findings of this study could be valuable for researchers and farmers who are working on plant segmentation from aerial images.

## U-Net
To segment pictures swiftly and precisely, U-Net, a convolutional neural network, is utilized. For model training, large datasets are often necessary. However, it is impossible to acquire the necessary quantity of information for image categorization. U-Net beats traditional models in terms of architecture and pixel-based image segmentation made from convolutional neural network layers. It functions well even with photographs from a tiny dataset [^1].
![unet](https://github.com/doadates/U-Net-and-Attention-U-Net/blob/main/images/unet.jpg)

## Attention Mechanism

In the context of image segmentation, attention mechanisms are used to selectively focus the model's attention on the most relevant regions of the image during training. 

Aiming to resemble cognitive attention, the attention strategy. To urge the network to pay greater attention to the little but significant portions of the data, the impact boosts some portions of the input data while weakening others. The essential idea is that when predicting an output, the model only considers the portions of the input string where the most crucial information is concentrated. In other words, it merely considers a small number of the input words [^2].

There are two main types of attention mechanisms: hard attention and soft attention. Hard attention involves cropping the image to focus on a specific region, but this approach is not feasible for our implementation. Therefore, we propose to use soft attention, where desired weights are assigned to the model based on ground truths or other methods. Specifically, in our research, we will use ground truths to assign these weights to the model, which will allow it to selectively focus on the most relevant regions of the image during training and improve the accuracy of the segmentation results.

Figure below shows the soft attention mechanism. Based on previous additional object localization models, localization and subsequent partitioning phases have been separated to improve partitioning performance. This can be achieved by adding attention gates on top of the U-Net architecture without training new models. Therefore, U-Net attention gates can improve model precision and accuracy for foreground pixels without adding too much processing overhead. Before combining, attention gates are used to aggregate only relevant activations. Gradients from the background are reduced during the back pass [^3].
![att_mec](https://github.com/doadates/U-Net-and-Attention-U-Net/blob/main/images/unet-attention.jpg)
## Attention U-Net

Through an interface that links the encoder and decoder, the decoder is able to access data from each hidden state of the encoder. This framework enables the model to concentrate on the input sequence's most crucial segments and then comprehend the relationships between them. The Attention-based U-Net CNN model is a variation of the popular U-Net architecture that incorporates attention mechanisms to selectively focus the model's attention on the most relevant regions of the image during training.

U-Net is a popular architecture used in semantic segmentation tasks, it's structure is composed by an encoder and a decoder, where the encoder is responsible to extract features and the decoder use these features to generate the segmentation map. The attention-based U-Net CNN model enhance the standard U-Net architecture by incorporating attention mechanisms in the form of gating mechanisms in the decoder part. These gating mechanisms assign desired weights to the features from the encoder, allowing the model to selectively focus on the most relevant regions of the image during training[^4].
![attention](https://github.com/doadates/U-Net-and-Attention-U-Net/blob/main/images/attention.jpg)

In our project, we propose to use this Attention-based U-Net CNN model as it has been shown to improve the performance of semantic segmentation tasks and it will allow us to selectively focus on the most relevant regions of the image during training, which will enhance the accuracy of the segmentation results.

## Requirements
The codes in Google Colab will be written in Python programming language. With respect to Python programming language usage, some important libraries for computer vision and data visualization can be provided to process the model. These Python libraries are;

[![TensorFlow 2.1](https://img.shields.io/badge/TensorFlow-2.1-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0)
[![NumPy](https://img.shields.io/badge/NumPy-1.21.0-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3.0-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.2-blue.svg)](https://matplotlib.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.3-blue.svg)](https://opencv.org/)
[![Keras](https://img.shields.io/badge/Keras-2.6.0-blue.svg)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-blue.svg)](https://scikit-learn.org/)

## Datasets
- [5100 Dataset](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation?datasetId=1650618&select=Forest+Segmented)
  
  ![data1](https://github.com/doadates/U-Net-and-Attention-U-Net/blob/main/images/858771_sat_83.jpg) ![data2](https://github.com/doadates/U-Net-and-Attention-U-Net/blob/main/images/861353_sat_72.jpg)
  
  First dataset is with image/mask size = 256 x 256 that consist of 5100 samples, for each picture it has readily prepared mask. 
- [HIGH-QUALITY DATASET](https://cloud.pix4d.com/dataset/911895/files/inputs?shareToken=5e81e91f-8a73-4201-81f4-748056fa0370)

  
  Second dataset is with image/mask size = 3648 x 5472 that consist of 101 samples, and this dataset doesn’t have own masks. So for this dataset, new masks will be included according to the thresholding technique that is chosen.
  
  ![data2](https://github.com/doadates/U-Net-and-Attention-U-Net/blob/main/images/IX-11-23119_0096_0237.JPG) ![data2](https://github.com/doadates/U-Net-and-Attention-U-Net/blob/main/images/IX-11-23119_0096_0304.jpg)


## Training Parameters
`model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-3), loss=iou_coef_loss, metrics=['accuracy', iou_coef])`

The dataset is split into 3 part, 70% training, 20% validation and 10% test in our project.

[^1]: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. (2015, May. 18). U-net: Convolutional Networks for Biomedical Image Segmentation. [Online]. Available: https://arxiv.org/abs/1505.04597
[^2]: Kate Loginova (2018, Jun. 22). Attention in NLP [Online]. Available: https://medium.com/@edloginova/attention-in-nlp-734c6fa9d983
[^3]: Hong Jing (2019). Biomedical Image Segmentation - Attention U-Net [Online]. Available: https://jinglescode.github.io/2019/12/08/biomedical-image-segmentation-u-net-attention/
[^4]: Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Rueckert, D. (2018). Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.
