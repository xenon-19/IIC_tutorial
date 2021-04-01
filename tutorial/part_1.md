# Part 1. Method overview

Each method begins with a problem statement. Here we focus on clustering and semi-supervised classification problems, so we need to define them before we proceed.   

## Clustering



The clustering problem suggests that the initial data have no labels. A clustering model takes the input data and tries to group it into several clusters. The result is the labels assigned to each sample (see Image 1).  The miracle of the clustering methods is that they require no prior knowledge of the data grouping, only the data itself. A major drawback is that they work completely on their own and sometimes can produce unexpected results.

 <center><img src="https://github.com/vandedok/IIC_tutorial/releases/download/v0.1/clustering.png" width=600px /></center>
 <center>Figure 1: Image clustering </center>
 <br></br>

There are many clustering algorithms like K-means or DBSCAN, which are great for structured data -- tables.  However they are not really useful for unstructured data, like images or time series. To perform clustering on such kind of data, more advanced methods are required.


In real life clustering algorithms are hard to validate, as there are no true labels available. However it's possible to benchmark a clustering method using labeled dataset. One can take something like MNIST, feed the images without labels, make the clustering algorithm to produce clustering labels and than compare the clustering labeling with a true ones. If these two labelings are close to each other up to some permutation, than the clustering algorithm have done a great job.  

Some clustering algorithms require the knowledge of a number of final cluster, while others manage to figure it out themselves. IIC in its basic setting belongs to a former group.

## Semi-supervised classification

Imagine, that you have a huge dataset with only a tiny part of it being labeled. Is it possible to train a model in this setting and achieve a decent performance?

It appears that the answer is: "yes, sometimes it's possible". One way to do so is to utilize a clustering algorithm on a whole dataset, both labeled and unlabeled parts. After a clustering labeling is obtained, one can use a labeled part to assign to each cluster a specific class (see Image 2). Using this approach one can reduce the task of weakly  supervised classification to clustering task.

<center>
<img src="https://github.com/vandedok/IIC_tutorial/releases/download/v0.1/weakly_class.png" width=600px />
</center>
<center>
Figure 2: Semi-supervised classification
</center>
<br></br>

The validation of weakly supervised classification can be done in a similar manner to a regular classification -- just split the labeled part into training and validation set and use the latter one to estimate the performance, but not in training. Sometimes the labeled part is so small, that  it makes no sense to split it. In this case one  train the model only on the unlabeled part (recall, that clustering model require no labels for training), than use labeled part both for class assignment and for performance estimation.




The class assignment step the latter approach is a data leak -- the final models answers are corrected with the help of a validation set, which isn't good. However, if we expect our clustering algorithm to have a decent performance, this leak has little effect and thus can be forgiven.

## Model structure

Ok, we understand, that for both tasks we need to cluster a bunch of unstructured data, say images.

How can we do it? First we need to set up or model.  In our image setting its structure  is a really basic one.  It consists of three consequential parts:

1) A fully-convolution __backbone__ (say a ResNet model without last fully-connected layer),

2) A flatten layer

3) A fully-connected layer, which we will aslo call a __clustering head__.

The latter one has a number of output features equal to a number of classes/clusters. Each output feature, or __logit__, represents the output cluster/class  The higher  the  value of the __logit__, the more the model has a tendency to assign the sample to a given class.

 If you've ever encountered any modern classification task on images, you've probably seen this structure. In different contexts it can be called as __encoder__ or __feature extractor__. To train such a model without external labels is not a straightforward task. To see, how IIC encounters it, proceed to next section.

 ## IIC forward run

A training procedure in DL is usually refers to optimization of some loss function by adjusting the model's weights. I think it's really instructive to look at IIC forward run step by step:


1) Take your dataset (or, more practically, a batch of images) and __augment__ it: perform a random transformations such as rotations, flipping, scaling and so on. The more varied the transformations the better. The only constraint is that after the transformation the object on the augmented image should still be recognizable . After this step you will have the initial images batch and a transformed images batch.

2) Apply the encoder to both batches, than use  a softmax function to convert the logits into probabilities. At the beginning of the training these probabilities s should differ a lot for initial and transformed  batches. However we understand, that the transformations should not affect the probabilities, as they do not change semantic meaning. If we find a way to make this distributions as similar as possible, we win.

3) Lucky to us, we do have a function which measures this difference -- it is called __mutual information__. Quoting the [wikipedia](https://en.wikipedia.org/wiki/Mutual_information), it "quantifies the 'amount of information', obtained about one random variable through observing the other random variable". I dedicated to the mutual information the  of this tutorial [second part](https://github.com/vandedok/IIC_tutorial/blob/master/tutorial/part_2.md). For now it's sufficient to know, that the mutual information is a differentiable function, which takes as input the cluster probabilities for all images in original and transformed batches.  


<center>
<img src="https://github.com/vandedok/IIC_tutorial/releases/download/v0.1/IIC_forward.png" width=600px />
</center>
<center>
    Figure 3: IIC forward run. x denotes original image,  gx -- a transformed image, dash line indicates, that the transformed and original images a processed with the sames CNN and and fully-connected layer.
    Based on Figure 2 from <a href="https://arxiv.org/abs/1807.06653">original paper</a>.
</center>
<br></br>

The intuition behind IIC is pretty simple -- the transforms should not change the meaning of the images, and thus the encoder should  give roughly the same output on initial and transformed image. To enforce the encoder to do so, we compute the mutual information between the outputs from initial and transformed images and optimize the encoder to make it as big as possible.

## Training and validation

At this point you can see, that IIC is based on a very simple intuition and perfectly fits in  a general deep learning framework. Two key components of it are well-chosen augmentations and the mutual information. Treating the mutual information as a loss function we can apply a backpropagation algorithm to obtain the gradients  and perform an optimization step, like in a simple classification setting. As this loss is differentiable with respect to model weights, you can utilize any gradient method you like to fit the model.

A good question is how to make a decision, where to stop the training and assume that the encoder is ready to produce the results. I don't have a general answer for that, however it's clear what's the simplest way to answer it: just wait when the loss function reaches plateau and state that there is the point you wanted to reach.

This methodology has an obvious drawback. Doing so you may find the local minima of the loss where the model behaves itself poorly. This is also affected by the fact that the performance of a DL model in general case is dependent on an initialization --  different runs made show you a noticeable divergence in the resulting performance.


For a clustering problem this problem can hardly be avoided and that's not the fault of IIC method but of the clustering setting itself. With the absence of true labels it's hard to say, weather the algorithm demonstrated a great performance. A possible workaround is to benchmark the algorithm on a labeled data and than use it on the unlabelled data of the similar structure.

For semi-supervised classification he validation procedure can be inherited from the classic classification setting: split the labeled set into validation and test subsets, use the validation one for label assignment and model selection and the test part for obtaining the final score.

## Conclusion

Now we had a look at the IIC approach and got an intuition how it works. The [second part](https://github.com/vandedok/IIC_tutorial/blob/master/tutorial/part_2.md) of the tutorial shows  what is mutual information in more detail and how can it be estimated. It contains a number of formulas -- if you don't like them, you may want to jump directly to the [third part](https://github.com/vandedok/IIC_tutorial/blob/master/tutorial/part_3.ipynb)  [![ Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vandedok/IIC_tutorial/blob/master/tutorial/part_3.ipynb) with the Pytorch implementation.

## Appendix: auxiliary overclustering



There are several tricks that can improve the performance of the IIC method. One of them is called __auxiliary overclustering__. The intuition is the following.

Imagine, that we want our data to produce a limited number of clusters, say N. In fact the data may have much more well-defined clusters which the model can actually recognize. Morover, sometimes it's easier for the model to train having in mind that abundance.

In IIC this fact can be utilized with the help of an additional fully-connected layer called __overcluster head__. This layer has the same number of input features as the last layer from initial IIC model and a number of output features several time larger. We replace original fully-connected layer with  overcluster head and train the model using mutual information as loss function (see Figure 4). This is done during several epochs, than  overcluster head is replaced  by original layer an so on.



<center>
<img src="https://github.com/vandedok/IIC_tutorial/releases/download/v0.1/IIC_overcluster.png" width=600px />
</center>
<center>
    Figure 4: IIC forward run with uxiliary overclustering .
    Taken from Figure 2 from <a href="https://arxiv.org/abs/1807.06653">original paper</a>

</center>
<br></br>

When the model is trained in with both clustering and overclustering heads, one can track down losses coming from both of them. When the overclustering head is training, clustering mutual information may not grow (or even decrease). The opposite is also true: during the clustering training overclustering mutual information may decrease. This training is illustrated on image 5.

<center>
<img src="https://github.com/vandedok/IIC_tutorial/releases/download/v0.2/IIC_overcluster_training.png" width=600px />
</center>
<center>
    Figure 5: IIC clustering and overclustering training</a>
</center>
<br></br>  


The <a href="https://arxiv.org/abs/1807.06653">original paper</a> (see table 2 there) states, that in some problems can greatly improve the performance of IIC.
