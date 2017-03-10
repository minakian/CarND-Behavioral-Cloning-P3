#**Behavioral Cloning** 

##Written Report V1

###This report is derived from the template provided.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/00Center.jpg "Center Lane"
[image2]: ./Images/01Start.jpg "Recovery Image Start"
[image3]: ./Images/02Recover.jpg "Recovery Image Recover"
[image4]: ./Images/03Corrected.jpg "Recovery Image Corrected"
[image5]: ./Images/04Reverse.jpg "Reverse Image"
[image6]: ./Images/05Reverse_Flipped.jpg "Flipped Reverse Image"
[image7]: ./Images/06Left.jpg "Left Image"
[image8]: ./Images/07Right.jpg "Right Image"


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
* video.mp4 as a video record of the car completing a lap of track 1

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I really like the architecture demonstrated in class by NVIDIA, but made a slight modification to reduce the chance of overfitting. The model consists of 5 convolutional layers, with ReLU activation, and 4 fully connected layers. The start of the model crops each image 70px off the top and 25px off the bottom. From here each image is normalized with a Keras lambda layer and given a zero mean.

The change is at line 60 where I added a 50% dropout layer. (I would like to experiment more with other architectures or modifications to this one, but I am behind in the class and once an acceptable model was achieved, I stopped. I will come back to this to improve my model though.)

####2. Attempts to reduce overfitting in the model

As stated above, a dropout layer was added to combat over fitting. I also separated out a validation set to verify if overfitting was occouring.

Other actions I took were: to record driving in both clockwise and counterclockwise directions, use both the left and right images, flip all images. This provided a much larger data set than just the traditional lap around the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and both left and right images.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a known architecture, the NVIDIA architecture, and modify as needed.

My first step was to implement a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it was demonstrated to perform well in self driving car applications.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it ran through fewer epochs.

Then I added a dropout layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle veered off the track, such as on the bridge and approaching the bridge. To improve the driving behavior in these cases, I recorded additional recovery data at different locations around the track shat seemed to provide unique scenarios.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3))
Lambda(lambda x: x / 255.0 - 0.5)
Convolution2D(24,5,5,subsample=(2,2),activation="relu")
Convolution2D(36,5,5,subsample=(2,2),activation="relu")
Dropout(0.5)
Convolution2D(48,5,5,subsample=(2,2),activation="relu")
Convolution2D(64,3,3,activation="relu")
Convolution2D(64,3,3,activation="relu")
Flatten()
Dense(100)
Dense(50)
Dense(10)
Dense(1)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle driving in the opposite direction to aleviate its tendancy to want to turn left and recorded recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself from undesired situations. These images show what a recovery looks like starting from the initial undesired point :

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would increase the training data available as well as generalize. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

Here are also examples of the left and right images used:

![alt text][image7]
![alt text][image8]


I then preprocessed this data by cropping and normalizing.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by a continued decrease in both training and validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
