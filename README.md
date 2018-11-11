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


---
## Detailed steps taken

#### 1. Building a FCN

Fully Convolutional Networks are especially designed for segmentation tasks, their structure being composed firstly of an encoder block, whose main function is to extract the features, which are going to be used by the second section, the decoder block. The decoder reconstructs the image to its original resolution, by using up-sampling and skip connection techniques. The number of encoding layers needs to be equal to the number of decoding layers.

In the implemented architecture, 3 encoder and 3 decoder layers have been chosen, as in the image below:

![image1]

#### 2. Choosing the learning hyper-parameters 

There are a number of parameters which can be modified and tuned for obtaining a better accuracy of the network, as it follows:
* `batch_size`: number of images that go through the network at once
* `num_epochs`: the number of times the dataset is passed through the network
* `steps_per_epoch`: number of batches that pass in 1 epoch
* `validation_steps`: number of batches that pass through the network in 1 epoch in the validation step
* `workers`: maximum number of used processes

The following hyper-parameters have been chosen for the training phase of the network, mostly by observing the results of previous trainig:
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

#### 3. Training the network

The model was trained in 25 epochs with GPU support, and it lasted approximately 2h30min.
The plot of the loss function over the epochs at three distinct time periods can be observed in the below images.

![image2]                  | ![image3]      		   | ![image3]
:-------------------------:|:-------------------------:|:-------------------------:
Epoch 4                    |  Epoch 14		           |  Epoch 24

#### 4. Predictions

The obtained model is firstly saved, and then further used to analyze its behavior on samples which belong to different scenarios possible to encounter.
In the tests below, the first image represents the input image as taken by the quadrotor, the second image represents the ground truth, the mask where the hero is labeled, whereas the third image is the actual output of the network.

![image5]

**Images while following the target**

![image6]

**Images without the target**

![image7]

**Images with the target in the distance**

As it can be observed, the trained network has deficiencies especially when the hero is situated at a certain distance. Segmentation in such situation can be further improved by training the network with more data where the hero is far away, data gathered from the simulator.

#### 4. Testing the network in simulation

Simulator was launched in **Follow Me!** mode and has first started to patrol until finding the target fo the first time.

![image8]                  | ![image9]      		  
:-------------------------:|:-------------------------:
Starting Simulator         |  Quad Patrolling		           

After finding the hero, it begins following her, the segmented image being as below.

![image10]                 | ![image11]      		  
:-------------------------:|:-------------------------:
Following Target           |  Input vs Output	     




### Final Score

The final scoring of the FCN is obtained by multiplying the final **Intersection over Union** with the **weight**, described as `weight = true_pos/(true_pos+false_neg+false_pos)`.

```
final_score = final_IoU * weight
```

The final score obtained is `0.419317475943`.

### Final Remarks and Comments

The network accuracy can be further improved, the first step being that of collecting new data to train on, but this can extend considerably the training process, depending on the dimension of the dataset.

The hyper-parameters are also in important factor and a better accuracy can be obtained by tuning them.





