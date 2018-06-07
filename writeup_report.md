# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/fidelity.jpg "Model Fidelity"


## Rubric Points

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 video of the test drive
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of:
1. normalization function
2. cropping function to only use relevant picture data
3. convolutional layer - depth: 24; kernel: 5x5; stride: 2
4. convolutional layer - depth: 36; kernel: 5x5; stride: 2
5. convolutional layer - depth: 48; kernel: 5x5; stride: 2
6. convolutional layer - depth: 64; kernel: 3x3; stride: 1
7. convolutional layer - depth: 64; kernel: 3x3; stride: 1
8. fully connected layer - output: 100
9. fully connected layer - output: 50
10. fully connected layer - output: 1

This scheme was derived from the NVIDIA model presented within the course.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 81). 


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 88).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approacha and final design

My first and final step was to use a convolution neural network model similar to the NVIDIA model presented in the course because it had already proven suitable within practical tests.

Since this model provided 10 channels as output values, an additional fully connected layer was added to only control the steerinh wheel angle.

To combat the overfitting, I modified the model so that an additional dropout after the convolutional layers was implemented. A dropout rate of 50% choosen to reduce the high overfitting tendency after the second epoch.

The final step was to run the simulator to see how well the car was driving around track one. There was a slight tendency of the vehicle to become unstable. This issue however was not caused by the model but by hardware issues on my laptop (high CPU usage). In the end this issue could be resolved by reducing the CPU ressources available for the simulator. Still a small instability can be observed at the end of the first round. The vehicle however is able to drive autonomously around the track without leaving the road.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to enter the center lane again after an offset.

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would increase the information on handeling right hand curves.

After the collection process, I had around 26000 elements of data points. To avoid memory issues, I utilized a generator for data akquisition.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. This picture illustrates the fitting process: 

![alt text][image1]

The ideal number of epochs was 3 after many trials with different numbers. I used an adam optimizer so that manually training the learning rate wasn't necessary.

