# Behavioral Cloning 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NetworkArch.PNG "Model Visualization"
[image2]: ./examples/driving_example.jpg "Grayscaling"
[image3]: ./examples/sim1.jpg "Recovery Image"
[image4]: ./examples/sim2.jpg "Recovery Image"
[image5]: ./examples/sim3.jpg "Recovery Image"
[image6]: ./examples/training_simulator.png "custom training sim"
[image7]: ./examples/ImageAugmentation.png "Image Aug"


## Rubric Points
##### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* [model.h5](https://github.com/tawnkramer/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network
* [video.mp4](https://github.com/tawnkramer/CarND-Behavioral-Cloning-P3/blob/master/video.mp4) a video of the final network controlling the vehicle in simulation 
* README.md this writeup summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

This architecture uses the 5 convolutional layers inspired by Nidia's seminal paper: [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). See model.py [make_model](https://github.com/tawnkramer/CarND-Behavioral-Cloning-P3/blob/master/model.py#L116) for an implementation in Keras. This uses 3 layers of 5x5 convolutions followed by 2 layers of 3x3 convolutions. Each layer used a stride of 2 to downsample the resolution of the output, as it added additional dimension for filtered output.

The model includes RELU layers to introduce nonlinearity [*](https://github.com/tawnkramer/CarND-Behavioral-Cloning-P3/blob/master/model.py#L116), and the data is normalized in the model using a Keras lambda layer. [*](https://github.com/tawnkramer/CarND-Behavioral-Cloning-P3/blob/master/model.py#L113)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. [*](https://github.com/tawnkramer/CarND-Behavioral-Cloning-P3/blob/master/model.py#L128)

The model was trained and validated on different data sets to ensure that the model was not overfitting [*](https://github.com/tawnkramer/CarND-Behavioral-Cloning-P3/blob/master/model.py#L214). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually [*](https://github.com/tawnkramer/CarND-Behavioral-Cloning-P3/blob/master/model.py#L140).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
For details about how I created the training data, see the next section. 

#### 5. Final result video
---
[![IMAGE ALT TEXT](https://img.youtube.com/vi/ZubjnJGXoiM/0.jpg)](https://www.youtube.com/watch?v=ZubjnJGXoiM "An Artificial Neural Network drives a car trained copy human driving.")

---

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from the accepted, capable NVidia design and then iterate on hyper parameters and layer settings to understand where improvements may lie. Multiple convolutional layers in this network are capable of creating higher layers of abstraction that allow for detection and use of lane edges and road features to predict steering. But they are equally capable of memorizing the course and using unrelated features such as tree location and sky features to predict steering. This manifests as low mean squared error on the training set, with much higher error on the validation sets.

One way of combating this tendency to memorize is to supply enough varied training data that the network is forced to learn abstractions. There is a balance between the quantity and quality of training data, and the network structure that will glean information from it. Neural networks tend to catasrophically forget. That is, they need equal samples of all possiblities at the same time. It could be trained to take one turn perfectly, and given a novel turn, fail to generalize. If then trained on only on the new situation, it would likely excell and then reduce it's fitness to handle the first case. Both cases must be sampled in sufficient ballance to create a network that can generalize to both.

Neural networks establish a deterministic function that will react predictably to the same input. Given the same exact image, the nework will produce the same answer each time, correct or not. It is incumbant on the creator and designer to make sure to sample enough possiblities to represent all possible cases it may see in the future. 

We are fortunate that driving is such a restricted domain that it is within the realm of possiblity to represent enough of the problem set to allow it to abstract in a useful way. This is only possible because of the redundancy and monotony of the driving problem. But it can not produce steering output any larger than it saw as input during training. The upper max of steering values is predetermined during training.

Consequently, the quantity of examples which require large steering values must be well represented in the training set. It is natural for most training sets to contain a predominance of examples with little or no steering. To combat that, I developed a [seperate training simulator](https://github.com/tawnkramer/sdsandbox) that can generate a large quantity of randomly generated road curvature in stochastically equal amounts. Further more, I used a PID controller with perfect knowledge of the road center to automatically generate a steering signal that was highly correlated with the road features. The tendency of PID controller to oscillate around the ideal path created an nice sampling of views and steering with very few straight or zero-steering samples.

Another way to force learning only relavant features is to mask areas of the image that are unlikely to contribute well, such as above the horizon line. This has the side effect of reducing the pixel count, which reduces the quantity of trainable weights in the network. Fewer weights equals faster training and acts to limit it's overall cabaility to over fit.

To combat the overfitting, I modified the model so that it used dropout layers at various points. This helps the network to learn redundantly, as it can't rely on a single neural connection. This is like have two or more experts rendering an opinion and then getting to choose between them. It does slow training in theory. But with modern GPU's, the delay is not noticed and more than made up for with a higher quality result that generalizes better. Dropout has largely replaced regularization as the prefered method for increasing generatlity and combating over-fitting.

Of course, the best way to prevent over fitting is to have a very large quantity of training data. The more the better. I've almost always seen model behaviors improve with significant additions to the training data set. And with the ai controlled PID driver, we have data in massive quantities.

Througout the testing process, the simulator was used to validate the performance on the final track. This simulator takes image data from a virtual camera on the car and send it to an external python process with our trained network. This network anaylyzes the image and produces a steering output. This steering is sent over the network connection to the simulator and is applied to the virtual steering wheel.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded more data from my custom simulator:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Here is a birds eye view to see one example of the curves created randomly. The road textures were taken from screen shots of the Udacity simulation in Unity from their [open source tree](https://github.com/udacity/self-driving-car-sim). The same vehicle was used and the camera coordinates matched.

![alt text][image6]

Then I repeated this process on track two in order to get more data points.

As we load images, we can optionally augment them in some manner that doesn't change their underlying meaning or features. This is a combination of brightness, contrast, sharpness, and color PIL image filters applied with random settings. Optionally a shadow image may be overlayed with some random rotation and opacity. We flip each image horizontally and supply it as a another sample with the steering negated.

![alt text][image7]

After the collection process, I had 120K data points. I experimented with image augmentation. But this yielded no appreciable improvement in performence in this simlulated environment. In real work scenerios, I have experienced that image augmentation greatly improvmes it's ability to generalize. But in this setting, it only slowed the training process and ultimately not used.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
