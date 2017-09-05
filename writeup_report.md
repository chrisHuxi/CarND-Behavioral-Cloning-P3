#**Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Reference to "NVIDIA Architecture" metioned in videos, my model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 91-95) 

The model includes RELU layers to introduce nonlinearity (code line 91-103), and the data is normalized in the model using a Keras lambda layer (code line 90). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 98-104). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 34). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 108).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and I tune angle correction to 0.2, 0.17, 0.15, 0.1, interesting, I find when I use 0.18 model get lowest validation loss, but the trained model can't run a perfect round in simulator, in fact when I use 0.1 as correction value I can get a not bad result.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convolution neural network model.

My first step was to use a convolution neural network model similar to the "NVIDIA Architecture" I thought this model might be appropriate and it might be a good start point because it was proved useful.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that apply dropout in model. concretely, first I apply dropout layer in each convolutional layer with dropout value = 0.4, and result is not just as my wishes, then I tune dropout value = 0.35, 0.3, 0.2, all this can not improve result. I guess maybe it is because convolutional layer aim to recongnize feature of images, so that it perform better when we let all information go through them. So that I apply dropout in fully connected layer, and tune dropout value = 0.3, then I get a better model.

Then I use flip and use images and angles from three cameras, so that we can get more data to train, the easiest and most useful way to combat the overfitting. More data is always better.

Besides I tune the epoch number to 10, because I find when I use 5 epoch, car always go aside and when it turns around it leaves the road likely. And when I do this adjust, our model performs better, even it will cost much longer time.

And I try to apply grayscale to image, still it didn't achieve better performence. So I give it up.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. 

  1) 
  
![alt text][image1]
  
  2) 
  
 ![alt text][image1]

To improve the driving behavior in these cases, I tune the epoch number, dropout value and angle correction. But those methods bring little effect. Then I find tune the speed value in drive.py is a good idea. After all, slower is always safer, the same as reality.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 88-105) consisted of a convolution neural network with the following layers and layer sizes. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Cropping2D         		| Crop 160x320x3 image to 160x250x3   							| 
| Lambda         		|    		normalize images					| 
| Convolution 5x5     	| 2x2 stride, valid padding, depth = 24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, depth = 36 	|
| RELU					|			
| Convolution 5x5     	| 2x2 stride, valid padding, depth = 48 	|
| RELU					|	
| Convolution 3x3     	| 1x1 stride, valid padding, depth = 64 	|
| RELU					|	
| Convolution 3x3     	| 1x1 stride, valid padding, depth = 64 	|
| RELU					|	
| Flatten	      	| 	|
| Fully connected		| hidden1 = 1164, hidden2 = 100, hidden3 = 50, hidden4 = 10, Output = 1, dropout = 0.3  |       					

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. But I find when I use data created by myself, our model always perform not as my wish. So I think it is because my driving skill is terrible. So that I download data from Udacity. 

the image looks like this:

![alt text][image1]

After the collection process, I had about 8,000 number of data points per camera. I then preprocessed this data by flip them and use data from all three cameras to get more data. Actually I try to apply grayscale in them, but it didn;t work. And also I use "Cropping2D" function and "Lambda" function to crop image and normalize them in model.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by test result. I used an adam optimizer so that manually training the learning rate wasn't necessary.
