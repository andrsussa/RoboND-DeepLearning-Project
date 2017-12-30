## Project: Follow Me

---

**Steps to complete the project:**

* Clone the project repo here
* Fill out the TODO's in the project code as mentioned here
* Optimize your network and hyper-parameters.
* Train your network and achieve an accuracy of 40% (0.40) using the Intersection over Union IoU metric which is final_grade_score at the bottom of your notebook.
* Make a brief writeup report summarizing why you made the choices you did in building the network.

[//]: # (Image References)

[image1]: ./misc/ENCODERSTAGE.png
[image2]: ./misc/ENCODERBLOCK.png
[image3]: ./misc/SEPARABLECONVO.png
[image4]: ./misc/1X1CONVO.png
[image5]: ./misc/DECODERSTAGE.png
[image6]: ./misc/UPSAMPLING.png
[image7]: ./misc/DECODERBLOCK.png
[image8]: ./misc/model.png
[image9]: ./misc/HYPERPARAM.png
[image12]: ./misc/HD5format.png
[image13]: ./misc/FINALIOU.png

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points

---

### Writeup / README

#### 1. Provide a write-up / README document including all rubric items addressed in a clear and concise manner. The document can be submitted either in either Markdown or a PDF format.

You're reading it!

#### 2. The write-up conveys an understanding of the network architecture.

The network is a Fully Convolutional Network which is comprised by a **Encoder Stage** formed by three encoding blocks and a 1x1
convolution layer, and a **Decoder Stage** formed by three decoding blocks.

**Encoder Stage**

The main objective for the Encoder Stage is to retrieve enough amount of features
from the input image for the classifier to perform effectively. This can be done through
several techniques, and in this case this is performed by implementing a series of convolutional networks in addition to a 1x1 convolutional layer, as seen in the image below.

![alt text][image1]

The reason for using convolutional layers instead of fully connected is because
the former preserves spatial information throughout the entire network rather
than flatten the tensor into 2 dimensions when passing it as input, and this
feature is aligned with the semantic segmentation funcionality that this project pursues.

Moreover, the specific technique used in this stage are Depthwise Separable Convolutions, which are what constitutes the "encoder_block" function. This type of convolution allows an increase of efficiency of the network, since it reduces the number of parameters in comparison to fully connected layers. This also has the extra perk of reducing overfitting for that same reason. The encoder block is shown in the image.

![alt text][image2]

The code that makes use of separable convolutions is seen in the following image. It takes
advantage of a high level deep learning API called Keras, which was used in several pieces of the project.

![alt text][image3]

Additionally, the use of a 1x1 convolutional layer at the end of this stage
allows the preservetion of spatial information further, since it practically is a matrix
multiplication with spatial information. Also, it makes the system capable of
receiving as input images of any size, rather than being fixed to only one. The
code used to implement it is printed next, in this case having a kernel size and
stride of 1, and a output layer depth varying depending on the task.

![alt text][image4]

It may have been noticeable that both the separable convolution and regular
convolution include a batch normalization step at their output. This is done so
a normalization is always included to the information that is being passed as
input to other layers within the network. Ultimately, this produces faster
training and high learning rates.


**Decoder Stage**

The Decoder Stage's goal is to scale back the information encoded previously into
the scale equal to the original. This results in obtaining a prediction of
each of the pixel that entered as input to the network. For this reason, the
number of decoder blocks will be same as the encoder blocks. This stage
implementation is seeing in the image.

![alt text][image5]

A decoder block is formed by three steps. First, an upsampling layer is
performed to bring the scale of the image back to its original. There exists
several methods to perform this task, and in the case of this project a bilinear
upsampling was chosen.

Bilinear upsampling is a resampling technique that uses the weighted average of neighboring pixels to estimate new pixels. It has the advantage of speeding up performance of the network despite the fact that it does not contribute to learning as other techniques like transposed convolutions do. Moreover, its application causes the lost of some details, hence an additional step is needed to improve on this matter. The code fo the bilinear upsampling function is seen below.

![alt text][image6]

Then, as a second step information from multiple resolution scales can be used to improve over
details lost on the upsampling and to get a better overall picture of the scene.
In our case, this is performed by concatenation since it allows us to combine
information from the upsampled layer and a previous layer with more spatial
details in an efficient way while having the advantage that the depth of these
two layers do not have to match.

Additional to this, as a third step some separable convolution layers are included to add on the
learning of the network by taking into account those particular spatial details included in previous layers.

By performing this steps the encoder block is formed as shown in the following image.

![alt text][image7]

An overall diagram of this whole process is shown below.

![alt text][image8]

**Parametrization**

Starting with the Encoder Stage, the optimization of this step was mainly dedicated to determining the number of
layers to have. Initially, I started by choosing two encoder blocks and then vary
the depth of the 1x1 convolution and test results. This resulted in a not
sufficient precision for the network and ultimately decided to upgrade to three
layers. Another important parameter to set in this stage is the size of the
depth after each block. Based on experience from previous labs, I decided to
follow a pattern of powers of 2, doubling its depth in each layer by applying a
stride of 2. Initially tested having 16 filters, and from there to
double the depth as layers were included. Finally settled with an initial
filter of 32 since it showed to have the best results.

Having decided this, the filters to use in the decoder stage were pretty straight
forward. To apply the Encoder-Decoder scheme it was imperative for the decoder
to follow a backwards utilization of the encoder filters, as was seen in the
decoder section implementation. Hence, the 1x1 convolution output was paired with the second
encoder layer, their output paired with the first encoder layer, and finally
their output paired with the input of the function.

Having settled on this configuration I could test the precision of the network
by varying the depth of the 1x1 convolution layer. Taking advantage of it being
able to manage variable sizes for its output, I was looking for a ideal balance
between efficiency and precision. Started by applying the same depth as the
final encoder block which was of size 128, obtaining positive results but not ideal.
Then went further to applying its half, 64, without achieving the minimal allowed score.
After some iterations I was satisfied with a filter size of 100 for the 1x1
convolution layer as can be seen in the model.

Finally, I want to mention that the data set used for training, validation and
testing was the one provided by the project and no further new data was used in
the model. This was due mainly because it was not necessary to achieve a minimum
score value but possibly is mandatory to obtain a better precision.

#### 3. The write-up conveys the student's understanding of the parameters chosen for the the neural network.

The hyperparameters chosen for this project can be seen below:

![alt text][image9]

The tuning of these parameters was mainly around the batch size and the number
of epochs. The other parameters were left as given, and the learning rate was
set based on experience of previous labs. It was mentioned that the
steps_per_epoch parameter could be determined based on the number of images and
the chosen batch_size, recommendation that at the end was not explored since the
results of the network were already satisfying.

The num_epochs parameter was determined based on the learning curves. Started by
having a value of 100, and based on progress and faster learning, that value
was iteratively decreased. It may be possible that the value set could be
decreased even further.

Regarding the batch size, it was varied in a similar fashion to the filters in
the encoder stage. Several values power of 2 were tested in combination with the
parameters of the network, deciding upon a batch size of 16 based on results.

#### 4. The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

***1 by 1 Convolution***

A one by one convolutions is when a convolution has a kernel size of 1x1 and
stride of 1. Additionally, a zero padding is also used for this configuration.

They are an inexpensive way of making a model deeper and to have more parameters without affecting the structure. Ultimately, they are just matrix multipliers and have the perk of needing fewer parameters.

They are used in Fully Convolutional Networks for semantic segmentation
tasks, since they provide the capability of preserving spatial information.
Also, since their layer size is not constrained like fully connected layers,
they are capable of giving the network the feature of receiving images of any
size.

***Fully Connected Layers***

In a fully connected layer the neurons have full connections to all activations in the
previous layer, and the activations can hence be computed with a matrix
multiplication followed by a bias offset
[1](http://cs231n.github.io/convolutional-networks/#fc).

They are commonly used as a way of learning non-linear combinations of features
normally coming from convolutioal layers. Hence, they are good for
classification tasks like asking about the presence or absence of a certain
object e.g. Is there a cat in the scene? 

#### 5. The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

Encoding an decoding is useful when we want to obtain a prediction or estimation
of each of the pixels of the input image. Apart from obtaining a rich set of
features while encoding, decoding allows us to reference the source of those
features for labelling purposes. That is the reason it is used in semantic
segmentation, since the goal there is to label each one of the pixels in the
scene into a specific category.

Problems that may be present with this technique is that depending on the depth
of the network or the set of learning and feature extraction parameters,
important details of the images may be lost and impossible to recover, since
upsampling is lossy by definition. This can be improved by adding skipping
connections and trying to complement information of the scene based on the
original images given as input.

#### 6. The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

As the structure of a neural network is important, is equally important to have a comprehensive data set. This data must cover several number of situations that the system may encounter. Not only including positive situation cases e.g. only pictures of the object to follow, but also the negative or absence situation e.g. scenes where the object is not present, in principle to avoid an overlly generalized model (overfitting). Also, it may be useful to identify particular cases where the model may have difficulties performing, and purposedly provide further data on the case e.g. in this project, it was often the case that the model failed when the hero was far from perspective and next to distractors, hence it was useful to focus efforts into expanding that case data set. Finally, it is worth mentioning that in this project's case, the object to follow (hero) had intentionally different features to similar objects (red color) which made the identification task easier, but possibly would not be so in a real world scenario.

### Model

#### 1. The model is submitted in the correct format.

The model file is called "model_weights" and is located in the path "data/weights/".
Its format is shown in the image:

![alt text][image12]

#### 2. The neural network must achieve a minimum level of accuracy for the network implemented.

The notebook "model_training.ipynb" is included in the repository and shows all
the code used to achieve a minimum level of accuracy of 40% using the
Intersection over Union metric. An image of the final steps can be seen below:

![alt text][image13]

