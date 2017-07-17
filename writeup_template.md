#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./pictures/center_2017_07_13_22_26_44_590.jpg "Center Steering"
[image3]: ./pictures/center_2017_07_17_23_40_39_928.jpg "Recovery Image 1"
[image4]: ./pictures/center_2017_07_17_23_40_40_883.jpg "Recovery Image 2"
[image5]: ./pictures/center_2017_07_17_23_40_42_389.jpg "Recovery Image 3"
[image6]: ./examples/center_2017_07_13_22_28_32_594.jpg "Normal Image"
[image7]: ./pictures/center_2017_07_13_22_28_32_594_flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_cs.h5
```
Note: I changed the drive.py, so that the car is driving with 32MPH instead of 9 MPH (drive.py: line 47)

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the NVIDIA model shown in the udacity teaching video, as it worked pretty good for driving the first track autonomously (see below line 55).

The model includes RELU layers to introduce nonlinearity (code line 61 ff.), and the data is normalized in the model using a Keras lambda layer (code line 59). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 67). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as well as laps that were driven using the mouse (smooth steering angles) and keyboard (rather "binary" angles).

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I was following along the training videos, but my car never achieved a worthy quality, meaning, it did not follow the track at all. It permanently drove to the left and fell of the track after about 3 seconds (with LeNet Model Architecture)

I then converted the images to greyscale and augmented the training data by mirroring the central cam images, and also took the side cam images into account (implementation as suggested in the teaching video). But the behavior of the car did not change much, it mostly steered to the right, going up the hill.

I then tried the nvidia model approach (as shown in the video), and the car instantly drove along the track without leaving it. Even changing the speed from 9 to 32 MPH did not lead to an instable controller of the car.


####2. Final Model Architecture

The final model architecture (model.py lines 58-71) corresponds to the model architecture developed by nvidia: it is a convolution neural network with 9 layers, incl. a normalization layer, 5 convolutional layers and 3 fully connected layers.


![NVidia Model][image1] (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving while steering with the mouse, and 2 laps while steering with the keyboard. Here is an example image of center lane driving:

![Center Steering][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer from the side of the road back to the center. These images show what a recovery looks like starting from ... :

![Off Track][image3]
![Going back to the middle][image4]
![Reached the center of the track][image5]

To augment the data sat, I also flipped images and used the side cam images. For example, here is an image that has then been flipped:

![Normal Image][image6]
![Flipped Image][image7]

After the collection process, I had 22129 number of data points. I then preprocessed this data by turning the images to greyscale, and flipping the center images horizontally.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by increase of the value of the loss function of the validation set after 4 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
