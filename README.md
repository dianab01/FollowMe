## Follow Me Project
### Final project implementation for the first term of Robotics Software Engineer Nanodegree
---
[//]: # (Image References)

[image1]: ./images/network_architecture.JPG
[image2]: ./images/training_epoch_4.JPG
[image3]: ./images/training_epoch_14.JPG
[image4]: ./images/training_epoch_24.JPG
[image5]: ./images/following_target.png
[image6]: ./images/no_target.png
[image7]: ./images/patrol_with_target.png
[image8]: ./images/sim_start.JPG
[image9]: ./images/quad_patrolling.JPG
[image10]: ./images/following_target.JPG
[image11]: ./images/segmented_image.JPG

# Purpose of the project

The purpose of this project is to implement and train a Fully Convolutional Network (FCN) that is used in solving the following segmentation scenario: a quadcopter is flying over a city environment, where various people are walking. From the images taken by the quadcopter, a target person (called *hero*) needs to be identified precisely in the frames and followed.


# Frameworks and packages used 
1. Anaconda managed virtual environment
2. Jupyter notebooks
3. Numpy, scipy
4. Python 3.5
5. Tensorflow 1.2.1, along with the Keras API

The trained network is tested within the Udacity quadrotor simulator, QuadSim, the communication between the Python module and simulator being socket-based.


# Fully Convolutional Networks - used techniques

Fully Convolutional Networks are especially designed for segmentation tasks, where objects need to be detected in images or in the frames of a video. The procedure is called semantic segmentation, which implies that the pixels in the original images need to be labeled as part of one of the classes to identify. In the current case, the labels are going to be either `hero`, `ordinary citizen` or `background`. The output of the networks consists of an image of the same size as the original image, each class being represented with a different color.

The architecture of an FCN consists, first of all, of an encoder block, a 1-by-1 convolution layer and a decoder block.

### Encoder

The encoder clock is formed by multiple convolution layers, its structure being the same as that of a normal Convolutional Neural Network. In fact, this initial section of the FCN can be a complex CNN architecture. The main purpose of the encoder block is to extract features, which are used by the decoder. 
**Separable convolutions** are applied, also known as **depthwise separable convolutions**, which means that the convolution kernel is applied over each channel of the input layer (spatial convolution), and then a 1x1 convolution combines the resulting feature maps into an only output layer (depthwise convolution). This technique is useful in the training process of Neural Nets since it considerably reduces the number of parameters of the model, thus increasing performance and reducing overfitting of the dataset, up to an extent. 
At the same time, the input to each layer is normalized (batch norm) using the mean and the variance of the values in the current mini-batch. This procedure enables a faster training of the network, since convergence can be achieved faster, but also the learning rates can be higher. As well, the normalization can be seen as a form of regularization, which deflects from reducing the entire network to a simple matrix multiplication (in a linear manner).

### 1-by-1 Convolutional Layer

Whereas in CNNs, after the convolutional layers, one or more fully connected layers would be employed, in FCNs they are replaced with 1x1 convolutional layers. Its main role is to maintain the spatial information, by outputting a 4D tensor, and not flatten it to a 2D tensor, which enables the future use of convolutional layers in the decoder.

### Decoder

The decoder block has an equal number of decoder layers as the encoder layers and it is used in order to reconstructs the image to its original resolution. Transposed convolutions (also known as **deconvolutions**) are applied in this step. The first technique used on the previous layers is the *bilinear upsampling**, which increases the dimension of the layer by using bilinear interpolation. It is implemented as a weighted average of (four) neighboring pixels, situated diagonally to the pixel whose value is wanted to be determined. The result is then utilized in the skip connection procedure, which concatenates the small current layer with a larger previous one (in this case from the encoding block, as it will be described shortly), whose separable convolution is then computed.

It is to be mentioned that the skip connections play a role in reconstructing the information which was lost throughout the convolution steps, which narrows the feature map to a few particular sections of the initial image. The combination of the two layers is represented by an element-wise addition, and the result is propagated to the next layer.


---
## Detailed steps taken

### 1. Building a FCN

#### Implemented Network Model

In the implemented architecture, 3 encoder and 3 decoder layers have been chosen, as in the image below:

![image1]

#### Encoder Block

The encoder block has been composed of three encoder layers, each having the stride of 2 and `same` padding, each being constructed through a separable convolution, as it follows:
```
def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

The first layer used a filter size of `32`, whereas for the next 2, the filter size is `64`. The chosen stride and padding mean that after each layer the image size is reduce by half.

#### 1x1 Convolution

The output of the encoder block is then input to the 1x1 convolutional layer, which implicitly uses a kernel size of 1, as well as a stride of 1, the chosen filter size being `64`.
```
conv_layer_batchnorm = conv2d_batchnorm(encoder_layer3, 64, kernel_size=1, strides=1 )
```

#### Decoder Block

The decoder block also consists of three decoder layers in order to rebuild the original image size. One such layer is formed by firstly applying the bilinear upsampling over a small layer, then concatenating it with a previous larger layer (the tensor dimension must match) and finally apply a separable convolutional layer.

```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    upsampled_small_ip_layer = bilinear_upsample(small_ip_layer)
    concatenated_layers = layers.concatenate([upsampled_small_ip_layer, large_ip_layer])
    
    separable_conv_layer1 = separable_conv2d_batchnorm(concatenated_layers, filters)
    output_layer = separable_conv2d_batchnorm(separable_conv_layer1, filters)
    
    return output_layer
```

As it can be observed in the network architecture image, the first decoder layer combines the output of 1x1 conv layer with the second encoder layer for the skip connection, the filter size being `64`. The second decoder layer combines the output of the first, with the first encoder layer, with the same filter size, whereas the third decoder layer uses the output of the latter, and the input image of the skip connection, with a filter size of 32.

In order to obtain the output image, a 2D convolution is applied on the output of the encoder block, which has a softmax activation function and `same` padding, the number of classes in the output being *three*, as stated above.


### 2. Choosing the learning hyper-parameters 

There are a number of parameters which can be modified and tuned for obtaining a better accuracy of the network, as it follows:
* `batch_size`: number of images that go through the network at once
* `num_epochs`: the number of times the dataset is passed through the network
* `steps_per_epoch`: number of batches that pass in 1 epoch
* `validation_steps`: number of batches that pass through the network in 1 epoch in the validation step
* `workers`: maximum number of used processes

The following hyper-parameters have been chosen for the training phase of the network, mostly by observing the results of previous training:
```
learning_rate = 0.001
batch_size = 100
num_epochs = 25
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

Normalizing inputs to every layer of the network allowed choosing a smaller learning rate.
At the same time, it has been observed that the current final score of the network, which is roughly over 0.4, could have been obtained only with more than 20 epochs, with the implicit dataset.

### 3. Training the network

The model was trained in 25 epochs with GPU support in the provided workspace, which was based on a *Nvidia Tesla K80*, and it lasted approximately 2h30min. 
The plot of the loss function over the epochs at three distinct time periods can be observed in the below images.

![image2]                  | ![image3]      		   | ![image3]
:-------------------------:|:-------------------------:|:-------------------------:
Epoch 4                    |  Epoch 14		           |  Epoch 24

Train data can be found [here](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip), validation data [here](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip) and evaluation data [here](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip) and need to be placed in the `/data/train`, `/data/validation` and `/data/sample_evaluation_data` respectively.

**Resulting weights and configuration file can be found in `/data/weights`.**

### 4. Predictions

The obtained model is firstly saved, and then further used to analyze its behavior on samples which belong to different scenarios possible to encounter.
In the tests below, the first image represents the input image as taken by the quadrotor, the second image represents the ground truth, the mask where the hero is labeled, whereas the third image is the actual output of the network.

![image5]

**Images while following the target**

![image6]

**Images without the target**

![image7]

**Images with the target in the distance**

As it can be observed, the trained network has deficiencies especially when the hero is situated at a certain distance. Segmentation in such situation can be further improved by training the network with more data where the hero is far away, data gathered from the simulator.

### 5. Testing the network in simulation

Simulator was launched in **Follow Me!** mode and has first started to patrol until finding the target for the first time.

![image8]                  | ![image9]      		  
:-------------------------:|:-------------------------:
Starting Simulator         |  Quad Patrolling		           

After finding the hero, it begins following her, the segmented image being as below.

![image10]                 | ![image11]      		  
:-------------------------:|:-------------------------:
Following Target           |  Input vs Output	     


### 6. Final Score

The final scoring of the FCN is obtained by multiplying the final **Intersection over Union** with the **weight**, described as `weight = true_pos/(true_pos+false_neg+false_pos)`.

```
final_score = final_IoU * weight
```

The final score obtained is `0.419317475943`.


---
## Future Enhancements and Comments

Even though the model was trained on recognizing people, it can be trained to recognize any object of interest, whether it is an animal or a household article. It is however necessary to have a large enough set of labeled data with that object.

The network accuracy can be further improved, the first step being that of collecting new data to train on, but this can extend considerably the training process, depending on the dimension of the dataset.

The hyper-parameters are also in important factor and a better accuracy can be obtained by tuning them.





