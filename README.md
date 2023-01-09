I INTRODUCTION

Image classification can be defined as the task of categorizing images into one or multiple predefined classes. Although the task of categorizing an image is instinctive and habitual to humans, it is much more challenging for an automated system to recognize and classify images. If we search for a tag "love" on google, we get a wide variety of images: roses, a mother holding her baby, images with hearts, etc. these images are very different from one another and yet depict the same emotion of “love” in them. In this project, I explored the possibility of using deep learning to predict the emotion depicted by those images. (Vasavi and Aditi). In 1970, Paul Ekman and Friesen defined six universal basic emotions: happiness, sadness, fear, disgust, surprise, and anger, such basic affective categories are sufficient to describe most of the emotions that has been exhibited. However instead of ‘anger’ I chose ‘violence’ as it is more suited to the applications we are looking at - more and more people are turning to social media to show their support for causes or protest against something (for example, a lot of social messages poured in after terrorist attacks in Paris in November, 2015) for the same reason, I add ‘love’ as an emotion category. To limit to 5 emotions, I dropped ‘disgust’ and ‘surprise’. So, the final chosen emotion categories are: love, happiness, violence, fear and sadness. After finalizing the categories, I collected data (images) for these categories from google. I experimented with various deep learning classification methods and finally tried transfer learning from a convnet with millions of parameters, which is pre-trained on large-scale data (ImageNet) for object recognition.

OBJECTIVES OF THE STUDY

1.	To manually curate a dataset that consists of diverse set of images that depicts the same emotion for a particular category.
2.	To construct a deep learning classifier that classifies the images based on the emotion depicted.
3.	To feed the images for neural network i.e., simple feed forward neural network, and convolution neural network.
4.	To conduct experiments with different variants of the same network by fine tuning 
5.	To compare the performance of the model with different variants and to select a best model from the available ones.

SCOPE OF THE STUDY

Emotion classification in images has applications in automatic tagging of images with emotional categories, automatically categorizing video sequences into genres like thriller, comedy, romance, etc (Vasavi and Aditi).
To understand the general sentiment of people about an event that is happening in the society. Understanding sentiments or emotions from images has different applications including image captioning opinion mining, advertisement and recommendation (Takahisa yamamoto et al., 2021).
Inferring emotion tags from the images from social media has great significance as it can play a vital role in recommendation systems, image retrieval, human behaviour analysis and, advertisement applications (Anam Manzoor et al., 2020).

CHALLENGES OF THE STUDY

The problem of labelling images with the emotion they depict is very subjective and can differ from person to person. Also, due to cultural or geographical differences some images might invoke a different emotion in different people - like in India people light candles to celebrate a festival called “Diwali”, however in western countries candles are lit, most of the times, to mark an occasion of mourning. In addition to the above, there is another fundamental challenge involved in tackling the problem of extracting emotional category from images. A person could have tagged an image as, say “fear”, it could be because the image makes them feel that emotion when they look at it or the person in the image may experience fear. This kind of image tagging can confuse the network (Vasavi and Aditi).


II METHODOLOGY

PROCESS FLOW OF THE PROJECT

Start the Process. Connect to the storage driver, where the data is stored. Load the images into Train, Validation and Test batches respectively. Define the network architecture i.e., add desired layers. Configure the model with defining the metrics for classification, optimizer and the loss function. Train or fit the model to the architecture. 
Print the wrongly classified images to understand, for which category of images the model is more confused. Check for the training and validation accuracy. Document the entire architecture defined, optimizer and loss functions used, no. of epochs trained with the received accuracy. If accuracy is lesser than 70% fine tune the architecture of the model i.e., Capacity of the network (No. of Dense layers) can be reduced or increased, the optimizer used can be changed, the learning rate in the optimizer can be adjusted, no. of epochs can be increased, data can be normalized, convolutional layers can be added and so on. 
Then, check if the model is underfitted or overfitted. If underfitted increasing the number of epochs is the solution. If overfitted, there are various methods to reduce overfitting such as weight regularization, dropout layer, image augmentation, reducing the learning rate, using the weights of the pretrained network and fine tuning the blocks of the pretrained network.
If the model is not overfitted or underfitted, then try increasing epochs to check whether there is a further scope for improvement. If there is an improvement in the accuracy, increase epochs or if the accuracy is stagnant, save the model and stop the process. 

IMAGE PRE-PROCESSING

When it comes to training images in deep learning, we have to consider these two things:
1. Datatype: 
Data should be of homogenous type i.e., dtype of train_images[0], train_images[1], train_images[2] should be same. Either all are “int32” or “float64” or “float32”. 
Preferable data type for neural network is float32
2. Range:
Pixel value of grayscale or rgb images ranges from 0 to 255. Working with this huge range of values may deteriorate the learning. To enhance learning, the range can be changed from 0 to 255 to 0 to 1. 

TECHNIQUES USED TO OVERCOME OVERFITTING 

•	Optimizing the capacity of the network
•	Weight Regularization
•	Dropout
•	Optimizing the learning rates
•	Adding convolutional layers
•	Image Augmentation
•	Transfer Learning
•	Fine tuning the pretrained network

Optimizing the capacity of the network:

1.	Units in the Dense layer can be decreased or increased
2.	Number of Dense layers can be added or removed
3.	Number of convolution layers can be added or removed

Optimizing the learning rates:

The optimizer that has been mentioned during model configuration has learning rate as a parameter. Decreasing learning rate may enhance learning and slow up the convergence, increasing leaning rate may speed up convergence and can lead to missing the optimum value.

Adding convolutional layers:

For image processing the convolutional neural network works well when compared to deep neural network. So, adding convolution layer to the network architecture can improve the performance of the model. 

Image Augmentation

Overfitting also happens when we have less data. So, increasing the number of data points can reduce overfitting. But, when we have very less chance of collecting new data, we can augment the existing images for generating additional training samples. For ex: Taking image1 and flipping it, rotating it, zooming it, increasing brightness, shearing it and generating 5 different images from the existing image. 

Transfer Learning

Extracting important features of the data and feeding it to the model improves performance of the model. Feature extraction can be done with the help of a pretrained network. Transfer Learning is the concept of using weights of the pretrained model to our existing model. A pre-trained model is a saved network that was previously trained on a large dataset, (for example – ImageNet) for classification.
The intuition behind transfer learning for image classification is that if a model is trained on a large and general enough dataset, this model will effectively serve as a generic model of the visual world. We can then take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset.

Every Pretrained network has two main things:

1.	Convolution Base
2.	Classifier
Convolution base contains blocks of convolution and pooling layers and the Classifier defines the classifying criteria.

Feature Extraction: 

Use the representations learned by a previous network to extract meaningful features from new samples. If we use our own classifier and use the convolution base of the pretrained network, we are actually using the representations learned by the previous network to extract meaningful features from new samples.  You do not need to (re)train the entire model. The base convolutional network already contains features that are generically useful for classifying pictures. 

Fine tuning the pretrained network:

Unfreeze a few of the top layers of a frozen model base and jointly train both the newly-added classifier layers and the last layers of the base model. This allows us to "fine-tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.

PRE-TRAINED NETWORKS USED

1. VGG16
	VGG stands for Visual Geometry Group; it is a standard deep Convolutional Neural Network (CNN) architecture with 16 convolutional layers. The VGG16 model achieves almost 92.7% top-5 test accuracy in ImageNet. ImageNet is a dataset consisting of more than 14 million images belonging to nearly 1000 classes. The network has a default image input size of 224 * 224
2. RESNET 50 
	ResNet stands for Residual Network. Trained on ImageNet. ResNet50 is used to denote the variant that can work with 50 neural network layers. Residual blocks make it considerably easier for the layers to learn identity functions. As a result, ResNet improves the efficiency of DNN with more neural layers while minimizing the percentage of errors. The network has a default image input size of 224 * 224
3. Xception
	Xception is a convolutional neural network that is 71 layers deep. It involves separable convolutions. Trained on ImageNet. The network has a default image input size of 299 * 299. 


III DATASET
DATA COLLECTION

I have collected 5000 images in total with 1000 images in every category for the selected 5 emotional categories - Love, Happiness, Violence, Fear, and Sadness from Google. I split the data in each category such that 73% of the data (3630) is used for training, and 27% of the data (270) is used for testing in DNN experiments and 80% for training and 20% for validation in CNN experiments.
Steps involved:
Step 1: Searching for the tags in google
Step 2: Downloading the images with the help of “Fatkun batch download” chrome extension
Search Keywords Used:
The thousand images in every class have been collected with the search term:
1.	“fear” – scary, scary places, ghosts, haunted places, haunting scenarios, anxious, fright, panic, terrific.
2.	“happiness” – laughter, happy employee, party, happy emojis, kids laughing, delighted, smiling. 
3.	“love” - Love, mom and kid, pet’s love, dad and kid, marriage, heart in, love emojis, affection. 
4.	“sadness” – sad, sorrow, depressed, stressed, upset, disappointment, heartbroken, pessimistic.  
5.	“violence” – violence, protest, right to freedom, crime, kill, fight, dispute.

Data cleaning:

Images that are irrelevant or not exactly depicts the emotion are ignored
The images that contain entirely texts or quotes are ignored 
Images that are not in the .jpg format are ignored. 

AVAILABLE DATASETS

S. N	Name	Total images	No. of Categories
1	Flickr & Instagram	23000	8
2	EmotionROI	1980	6
3 	Art Photo	806	8
4 	IAPSa	395	8
5 	Stock Emotion	1.17 million	8
6	Abstract	228	8
7	Twitter I	1269	2
8	Twitter II	603	2
9	Flickr	60745	2
10	Instagram	42,856	2

The datasets that have eight categories mostly has Amusement, Awe, Anger, Contentment, Disgust, Excitement, Fear and Sad as categories
The datasets that have six categories mostly has Anger, Disgust, Fear, Joy, Sad, Surprise as categories

CONCLUSION:

The results show that deep learning does provide promising results on the emotion classification task. Emotion classification in images has applications in automatic tagging, automatically categorizing video sequences into genres like thriller, comedy, romance. While using Deep Neural Network alone, the accuracy of the model was very low.  After normalising the data, the accuracy improved. After adding convolution layers the accuracy of the model improved drastically. After augmenting images, the model further improved. Increasing learning rate diminishes learning and decreasing learning rate enhances learning of the network. Adding any one type of weight regularisation is sufficient for the data, adding both decreases the performance. Too many units in the dense layer decreases the performance of the model for this particular dataset. Adadelta optimiser is not well suited for this dataset. SGD optimiser is better than both Adam and RMSProp optimisers for this dataset. After using “Xception pretrained network” for feature extraction, the model performed comparatively well with an accuracy of 74.68%. Compared with the baseline model (6.2.1 Experiment 1) the accuracy of model improved from 20% to 74%.

IV REFERENCES

1.	Anam Manzoor, Waqar Ahmad, Muhammad Ehatisham-ul-Haq, Abdul Hannan, Muhammad Asif Khan, M. Usman Ashraf, Ahmed M. Alghamdi and Ahmed S. Alfakeeh (2020) “Inferring Emotion Tags from Object Images Using Convolutional Neural Network”
2.	Jingyuan yang, Jie Li, Xiumei Wang, Yuxuan Ding and Xinbo Gao (2021) “Stimuli-aware visual emotion analysis.”
3.	Jufeng yang, Dongyu She, Yu-Kun Lai, Paul L. Rosin, Ming-Hsuan Yang “Weakly supervised coupled networks for visual sentiment analysis.”
4.	Lailatul qadri binti zakariaa, Paul Lewis, Wendy Hallc (2017) “Automatic image tagging by using image content analysis.”
5.	Minyoung huh, Pulkit Agrawal, Alexei A. Efros (2016) “What makes imagenet good for transfer learning?” 
6.	Ms. Sonali b Gaikwad, prof. S. R. Durugkar (2017) “Image sentiment analysis using different methods: a recent survey”
7.	Quanzeng you, Jiebo luo, Hailin jin, Jianchao yang (2016) “Building a large scales dataset for image emotion recognition: the fine print and the benchmark.”
8.	Rameswar panda, Jianming Zhang, Haoxiang Li, Joon-Young Lee, Xin Lu, and Amit K. Roy-Chowdhury “Contemplating visual emotions: understanding and overcoming dataset bias (supplementary material).”
9.	Shaojing fan, Zhiqi Shen, Ming Jiang, Bryan L. Koenig, Juan Xu, Mohan S. Kankanhalli, and Qi Zhao “Emotional attention: a study of image sentiment and visual attention.” 
10.	Takahisa yamamoto, Shiki takeuchi, and atsushi nakazawa (2021) “Image emotion recognition using visual and semantic features reflecting emotional and similar objects.”
11.	Vasavi gajarla, Aditi gupta “Emotion detection and sentiment analysis of images.”
12.	Xu can, et al (2014) “Visual sentiment prediction with deep convolutional neural networks.”
13.	You, quanzeng, et al (2015). “Robust image sentiment analysis using progressively trained and domain transferred deep networks.”
14.	Yue zhang, Wanying Ding, Ran Xu and Xiaohua “Visual emotion representation learning via emotion-aware pre-training.”
15.	Zijun wei, Jianming Zhang, Zhe Lin, Joon-Young Lee, Niranjan Balasubramanian, Minh Hoai (2020) “Learning visual emotion representations from web data.”
16.	https://www.intel.in/content/www/in/en/products/docs/processors/what-is-a-gpu.html#:~:text=GPUs%20may%20be%20integrated%20into,as%20a%20discrete%20hardware%20unit.
17.	https://research.google.com/colaboratory/faq.html#:~:text=Colaboratory%2C%20or%20%E2%80%9CColab%E2%80%9D%20for,learning%2C%20data%20analysis%20and%20education.
18.	https://www.python.org/doc/essays/blurb/ 
19.	https://www.ibm.com/in-en/cloud/learn/deep-learning 
20.	https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Architecture/feedforward.html 
