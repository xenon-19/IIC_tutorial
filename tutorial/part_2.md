# Part 2: IIC loss function

## About mutual information

Now let's discuss what is __mutual information__ and how can it be estimated.

Mutual information is a functional which takes as input a joint probability distribution of two random variables. That means, the for a pair of random variables it takes as input the probabilities for all pairs of values and outputs a number.

Before we start digging further let's recall the notion of __joint probabilities__. Denote the probability of random variables _A_ and _B_  to take values _a_ and _b_ simultaneously as _p_<sub>AB</sub> (_a_,_b_). Than the set of values _p<sub>AB</sub>_ (_a_,_b_) for all possible _a_ and _b_  is called a __joint probability  distribution__.

For our case it's sufficient to assume, that the random variables are discrete. Let's also assume, that the variables _A_ and _B_ can take the values 0, 1 .. _N_<sub>C</sub>.


The next notion to remember is the __marginal probability distributions__ . For random variable _A_ it is defined as:

<center>
<img src="https://render.githubusercontent.com/render/math?math=p_{A}(a) = \displaystyle \sum_{b=0}^{N_C} p_{AB}(a,b)">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>

The marginal probability distribution of _A_ tells us, how often _A_ is going to take this or that value, if we are not looking at _B_ at all.
For _B_ marginal probability distribution is defined in a similar manner:

<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle p_{B}(b) = \sum_{a=0}^{N_C} p_{AB}(a,b)">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>

Now we are ready to define a mutual information. It is given by a formula:
<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle I(A,B) = \sum_{a=0}^{N_C} \sum_{b = 0}^{N_C} p_{AB}(a, b) \log \frac{p_{AB}(a, b)}{ p_A(a) p_B(a)}">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>


The formula seems complicated, but the motivation is simple. To see it let's consider several properties of mutual information:

1) Mutual information is symmetric:  _I_(_A_, _B_) = _I_(_B_,_A_)

1) Mutual information is zero if and only if _A_ and _B_ are independent.  This makes perfect sense, since independent variables should not carry any information about each other.

2) Mutual information is also not less than zero. Together with the previous property, this means that the mutual information of dependent variables is often larger than of independent ones.

4) For a given random variable _A_ mutual information is maximized when _A_ is completely defined by  _B_ in a non-random manner.  Consequently, variables that completely define each other maximize mutual information.


<details><summary>Explanation of the fourth property </summary>
<p>

The mutual information can be expressed with the help of [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) and [conditional entropy](https://en.wikipedia.org/wiki/Conditional_entropy) as:
<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle I(A,B)  = H(A) - H(A|B)">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>    

If _A_ is a  random variable with a given distribution, _H_(_A_) is constant. Conditional entropy _H_(_A_|_B_) is always non-negative. Moreover, if _A_ is defined by _B_ (which means that after measuring _B_ we know exactly which value takes _A_), than _H_(_A_|_B_) = 0, thus in this case _I_(_A_,_B_) takes its maximum value equal to _H_(_B_).
</p>
</details>

Now it's clear why mutual information is a good measure of dependence of two random variables. The more interconnected they are, the bigger the quantity and vice versa.

## Mutual information estimation

As you can see from above, to calculate mutual information it's sufficient to estimate a joint probability function p<sub>AB</sub> (_a_,_b_). After this it's straightforward to calculate the marginal distributions and the mutual information itself. Before showing you how joint probabilities are estimated in IIC, I want to make a disclaimer: the formulae below should  be taken not as an absolute truth, but rather a model assumption.  

Let's denote _i_-th data sample as _x_<sub>_i_</sub>, the corresponding transformed data sample as _gx_<sub>_i_</sub> and the encoder function as _&Phi;_( . ). After the encoder processes the sample it gives a set of _N_<sub>C</sub> numbers (_N_<sub>C</sub> is equal to 10 for MNIST). We denote the entire set as _&Phi;_(_x_<sub>_i_</sub>)  for original data and  _&Phi;_(_gx_<sub>_i_</sub>) for  transformed data :

<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle \Phi(x_i) = \begin{pmatrix} \Phi_0(x_i) \\ \Phi_1(x_i) \\ \vdots \\ \Phi_{N_C}(x_i) \end{pmatrix} \qquad ">
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle \Phi(gx_i) = \begin{pmatrix} \Phi_0(gx_i) \\ \Phi_1(gx_i) \\ \vdots \\ \Phi_{N_C}(gx_i) \end{pmatrix}">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>   


The parameter _N_<sub>C</sub>  here has a clear meaning -- it's a number of cluster/classes in out problem.

We treat these vectors as probability distributions, conditioned on the _i_-th sample:
<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle P_{orig}(a|i) = \Phi_a(x_i) \qquad P_{trans}(a|i) = \Phi_a(gx_i)">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>    


To get the unconditional joint probabilities, we need to multiply the probabilities and to average over the data:

<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle P_{orig, trans}(a,b) = \frac{ \sum_{i \in batch}  \Phi_a(x_i) \Phi_b(gx_i)}{N_B}">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>  


where _N_<sub>B</sub> is a batch size.

<details><summary>For the watchful ones</summary>

 The multiplication step is not clear as it implies that the conditional probabilities _&Phi;_(_x_<sub>_i_</sub>)  and  _&Phi;_(_gx_<sub>_i_</sub>)  correspond to the independent random variables, which is not intuitive. To defend it one may say, that even if the conditioned random variables are independent, they can lead to correct distributions when only the "desired" class has high probability.  Perhaps we should treat that step as an educated guess and not to insist on a complete mathematical rigor here. If you know, how to justify it or how to treat it in a better way, contact me please.
</details>
<b style="word-space:2em">&nbsp;&nbsp;</b>    

In IIC setting we expect the joint probability to be a symmetric function: the probability to observe the pair of classes _(a,b)_ should be the same as the probability to observe the pair _(b,a)_. However in general case the above formula for the joint probabilities doesn't have this symmetry.  To fix the situation we perform a symmetrization:

<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle P^{sym}_{orig, trans}(a,b) = \frac{1}{2}\left(P_{orig, trans}(a,b) %2B P_{orig, trans}(b,a) \right)">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>  


We will omit the upper index "_sym_" further on.


Now the "vague model assumptions" part is over. After obtaining joint probability distribution it's straightforward to calculate the marginals:

<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle P_{orig}(a) = \sum_{b=0}^{N_C}P_{orig,trans}(a,b) \qquad P_{trans}(b) = \sum_{a=0}^{N_C}P_{orig,trans}(a,b)">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>  


And the mutual information:

<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle I (orig, trans) = \sum_{a=0}^{N_C} \sum_{b = 0}^{N_C} P_{orig, trans}(a,b) \log \frac{P_{orig, trans}(a,b)} {P_{orig} (a) P_{trans} (b)}">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>  


Our encoder _&Phi;_( . )  has a huge number of parameters, adjusting which we can try to make this function as large as we can. This expression is a differentiable function and thus can be optimized with any gradient method you prefer.

## Entropy vs degeneracy

Sometimes in IIC setting the model tend to group the data to a number of clusters much less than the desired. The worst case of this scenario is when all the samples get into one cluster. This situation is called clustering degeneracy and appears not only in IIC, but in other clustering methods.

Sometimes it possible to fight it by slightly modifying the mutual information. To see, how can it be done, let's recall a concept of __entorpy__. The [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) _H(A)_ of random variable _A_ is the defined as:

<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle H(A) = -\sum_{a=0}^{N_C} p_A(a) \log(p_A(a))">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>  


The entropy  can be seen as a measure of disorder of a random variable. It is maximized when all the samples are distributed equally within the clusters/classes.

The [conditional entropy](https://en.wikipedia.org/wiki/Conditional_entropy) _H(A|B)_ is defined as:
<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle H(A|B) = -\sum_{a=0}^{N_C} \sum_{b=0}^{N_C} p_{AB}(a,b) \log\frac{p_{AB}(a,b)}{p_B(b)}">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>  

It is minimized when the measurement of _B_ completely predicts the measurement of _A_.

Using the entropies the mutual information can be expressed as:
<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle I(A,B) = \frac{1}{2}\big( H(A) %2B H(B) \big) - \frac{1}{2}\big(H(A|B) %2B H(B|A)\big)">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>  

The optimization of mutual information can be seen as a fight between marginal entropies with conditional entropies. If all samples tend to group more intensely than required, we may want to  help the first term by adding the additional weight to it. So the modification of the mutual information should be:

<center>
<img src="https://render.githubusercontent.com/render/math?math=\displaystyle I_\lambda(A,B) = I(A,B) %2B(\lambda - 1 )\big(H(A) %2B H(B)\big)">
</center>
<b style="word-space:2em">&nbsp;&nbsp;</b>  


When the _&lambda;_ is equal to 1, the modified mutual information is equal to the original mutual information. Making _&lambda;_ greater than one pushes the model towards spreading the labels to different clusters.

## Conclusion

Wuf! We have finished to study mutual inforamtion loss. It's time to switch to practice -- to the [Part 3](https://github.com/vandedok/IIC_tutorial/blob/master/tutorial/part_3.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vandedok/IIC_tutorial/blob/master/tutorial/part_3.ipynb).


```python

```
