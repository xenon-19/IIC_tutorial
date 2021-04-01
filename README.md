
<center><img src="https://github.com/vandedok/IIC_tutorial/releases/download/v0.3/cover.png" width=800px /></center>


## Introduction

Hello everyone!

This tutorial is dedicated to  a beautiful technique called "Invariant Information Clustering" (IIC) introduced [here](https://arxiv.org/abs/1807.06653). With its help one can can encounter many tasks, such as:


* Image (and other data types) clustering
* Unsupervised image segmentation
* Weakly supervised classification
* Encoder training
* and many others ...

According to [paperswithcode.com](https://paperswithcode.com/paper/invariant-information-distillation-for), this method showed several SOTA results, therefore I believe it's worth considering.


This tutorial covers a very simple setting this technique is applicable to -- the clustering and weakly-supervised classification of MNIST. However, after solving this problem  it's  easier handle more complicated tasks and focus on the details, but not on the method itself.

This tutorial consits of three parts. The first part is a method overview -- here we state the problems, refresh some general DL concepts and see how IIC approach is organized. If you are willing to cover this topic in depth or see the applications to other problems, i.e. unsupervised segmentation, it's highly recommended to read the [original paper](https://arxiv.org/abs/1807.06653) or to visit [official repo](https://github.com/xu-ji/IIC).

The second part describes a notion of mutual information and shows, how can it be estimated in IIC method. To enable TeX formulas it's written in Jupyter notebook. You can open it both through github or through Colab, but the latter one is better in rendering TeX .

The third part shows the implementation and focuses on applied nuances. It is also made in Jupiter, so you can run it in colab or locally and try IIC yourself.

## Contents:

* Part 1: [Method overview](https://github.com/vandedok/IIC_tutorial/blob/master/tutorial/part_1.md)
* Part 2: [IIC loss function](https://github.com/vandedok/IIC_tutorial/blob/master/tutorial/part_2.md)
* Part 3: [Practice](https://github.com/vandedok/IIC_tutorial/blob/master/tutorial/part_3.ipynb)
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vandedok/IIC_tutorial/blob/master/tutorial/part_3.ipynb)

 ## Local installation

 If you are running part 3 locally, it's better to install the libraries using the  ```requirements.txt``` file. It can be done with the following line:

 ```sh
pip install -r requirements.txt
```

After this restart the jupyter runtime and begin exploring.
