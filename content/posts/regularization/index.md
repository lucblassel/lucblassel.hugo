---
layout: post
title: "How can we make linear regression better? Regularization."
date: 2020-02-10 10:00:00
excerpt_separator: <!--more-->
feature: https://upload.wikimedia.org/wikipedia/commons/e/ed/Residuals_for_Linear_Regression_Fit.png
latex: true
tags:
  - machine-learning
  - code
  - python
  - regression
  - supervized
  - linear-model
  - algorithmics
---

If you haven't read my post on linear regression I invite you to do so [here]({{< relref "/posts/implementing-linear-regression" >}}), but basically it is a method for modelling the relationship between variables \\(X_i\\) and a target feature \\(y\\) in a linear model. This modelling is done through learning weights \\(\theta_i\\) for each \\(X_i\\) supposing that our model looks something like this:

<!--more-->
$$
y = \theta_0 + \sum_{i=1}^n\theta_i\cdot X_i
$$

These weights are learned using gradient descent, if you don't know how that works please read the [post]({{< relref "/posts/implementing-linear-regression" >}}).  

Linear regression is a simple method that can lead to pretty good results in many cases, however when the nunber of features grows it can overfit (i.e. learn the training data and not actually generalize), this is where regularization comes in. 

# What is regularization ?

As I said above, when the number of feature grows or if you let the model train too long, there can be [overfitting](https://en.wikipedia.org/wiki/Overfitting). Overfitting is when your model learns the training data by heart, this means that the training error is very very low as the model has learned where every training point is, however if you test your model on a holdout sample *(i.e. a training example that the model has not seen before)* the prediction will be very bad. The model is bad at generalizing, it has just memorized the training data. This of course means our model is useless for predicting anything that is not in the training data, which of course makes it useless to us.  

If we take the following image *(taken from [wikipedia](https://en.wikipedia.org/wiki/Regularization_(mathematics)))* as example, the green curve represents the undelying data generating function, essentially what we want our model to learn, any unseen points that we want our model to predict will fall on that line. The red points represent training examples from this function. The blue line represents the function our model has learned. As we can see, both cuvres meet in the training examples, this means that the blue function will have a training loss of 0, which is great! however it is absolutely useless for predicting new points from the green curve. 

![Regularization](Regularization.svg#center)

In the linear model, this overfitting often manifests by the model learning very large weights in order to get a loss of 0. By adding the value of these weights in the cost function, the gradient descent algorithm will try to keep the weight values as small as possible, which means the model will be "smoother" and that usually leads to more generalizable models.

# OK, so whats the math behind this ?

## Norms and regularization
The two main regularization methods for linear regression are L1 regularization (also called LASSO), and L2 regularization (also called ridge). These regularization methods are based on [L1 and L2 norms](https://en.wikipedia.org/wiki/Norm_(mathematics)). If you don't know what that is, a norm is a function that takes a vector and spits out a real number > 0, you can think of it as the "length" of the vector.  

The simplest norm that you can think of is simply adding all the components of the vector, and since the norm must be non-negative we can just take the absolute value of each component, this is called the Taxicab or Manhattan norm. So if we have a vector \\(x\\) with \\(n\\) components:  

$$
taxicab = \sum_{i=1}^n |x_i|
$$

Another very useful norm is the euclidean norm, where we take the square root of the sum of squared components of the vector. This gives us:  

$$
euclidean = \sqrt{\sum_{i=1}^n x^2_i}
$$

If this rings a bell its normal, if we have a two dimensional vector, this euclidean norm is simply a reflection of the pythagorean theorem. Where the "length" of the vector is the hypothenuse of the right triangle formed by the \\(x\\) and \\(y\\) components.  

You may be asking yourself OK but what is the L1 or the L2 norm. Well in general the Lp norm of \\(x\\) is noted \\(||x||_p\\) and is calculated as follows:  

{{<longmath>}}
$$
||x||_p = \bigg( \sum_{i=1}^n x_i^p \bigg)^{1/p}
$$
{{</longmath>}}

So the L2 norm is:  

{{<longmath>}}
$$
||x||_2 = \bigg( \sum_{i=1}^n x_i^2 \bigg)^{1/2} = \sqrt{\sum_{i=1}^n x^2_i}
$$
{{</longmath>}}

which, as the keen-eyed among you have already noticed, is simply the euclidean norm. And the L1 norm is simply the taxicab norm.

## How does this fit in with linear regression ? 
In effect regularization is just adding the norm of the weights vector *(except fot he bias term \\(~\theta_0\\))* to the cost function. In this article I will only cover the L2 norm but the process is quite similar with the L1 norm.  

OK, from now on I will assume that you have read the [previous post]({{< relref "/posts/implementing-linear-regression#how-do-we-compute-theta-values-" >}}), in which the math for the unregularized linear regression is explained and derived, and the notation is introduced.  

We had derived the following cost function \\(C\\):

$$
C = \frac{1}{2m} \sum_{i=1}^m(\hat{y}^{(i)} - y^{(i)})^2
$$

Our new cost function for the ridge regression (L2 regularized regression), is simply the old one to which we add the squared L2 norm *(no square root)* of the weights vector multiplied by a regularization strength factor \\(\lambda\\). If we set \\(\lambda=0\\) we simply get the unregularized cost function.

$$
C = \frac{1}{2m}\bigg( \sum_{i=1}^m(\hat{y}^{(i)} - y^{(i)})^2 + \lambda\sum_{i=1}^n\theta_i^2\bigg)
$$

For gradient descent we need the derivative of this new cost function, fortunately since we simply added a new term, ther new derivative is equal to the old one plus the derivative of the new term. Even better, since the regularization term is a sum itself, the partial derivative is super simple to compute: 

{{<longmath>}}
$$
\begin{align*}
L_2 &= \lambda\sum_{i=1}^n\theta_i^2 \\
\frac{\delta L_2}{\delta\theta_k} &= 2\lambda\theta_k
\end{align*}
$$
{{</longmath>}}

which gives us the following partial derivative for the cost function *(the 2 disappears because the whole cost function is divided by \\(2m\\))*:  

$$
\frac{\delta C}{\delta\theta_k} = \frac{1}{m}\bigg(\sum_{i=1}^m(\hat{y}^{(i)} - y^{(i)})\cdot x^{(i)}_k + \lambda\theta_k\bigg)
$$

For the vectorial notation, since our regularization will only apply to the weights and not the bias term, I will introduce the vector \\(\Theta_x\\) which is the vector of feature weights (basically \\(\Theta\\) without \\(\theta_0\\)). Therefore our cost function, and the associated gradient can be written as follows:  

{{<longmath>}}
$$
\begin{align*}
C &= \frac{1}{2m}\bigg(\sum(X\cdot\Theta - y)^{\circ2} + \lambda\sum\Theta_x^{\circ2}\bigg) \\
\nabla C &= \frac{1}{m}\big(X^T\cdot(X\cdot\Theta - y) + \lambda\Theta_x\big)
\end{align*}
$$
{{</longmath>}}

Pretty painless no ? As we shall see it's also super easy to add regularization to our python implementation.
# How do we implement it in Python ? 

We only need to change 2 functions: `compute_cost` and `compute_gradient`. In order to change compute cost we need to add the L2 term to the cost function, multiplied by the \\(\lambda\\) regularization strength (which will be an input parameter). This is simple:  

```python
def compute_cost(theta, X, y, lambda_):
    l2 = lambda_ * sum(theta[1:] ** 2) / (2 * len(y))
    return sum((X @ theta - y) ** 2) / (2 * len(y)) + l2
```

You will rememeber that the `@` operator is the `numpy` matrix multiplication operator, and for the regularization term we sum `theta` weight vector excluding the 1st element because we do not want to regularize the bias term.  
Then we just need to update the `compute_gradient` function to add the gradient vector (with a leading 0 to fill up the bias term, that we do not want to penalize):   

```python
def compute_gradient(theta, X, y, lambda_):
    l2 = np.append([0], ((lambda_ * theta[1:]) / len(y)))
    return (X.T @ (X @ theta - y)) / len(y) + l2
```

Finally just update the function that does the gradient descent so that it can be called with the \\(\lambda\\) parameter:

```python
def gradient_descent(X, y, theta, alpha, max_iters, lambda_=0):
    history = []
    for _ in range(max_iters):
        theta = theta - alpha * compute_gradient(theta, X, y, lambda_)
        history.append(compute_cost(theta, X, y, lambda_))
    return theta, history
```

And that's it, you now have a regularized linear regression, *easy peasy*. 

## How does it compare to scikit-learn's version ? 

In order to get an idea of how our home-grown model is performing I evaluated it in the same way as the previous post, that is to say I split the Boston housing dataset into a training and testing set, normalized features according to the mean and standard deviation computed on the training set.  
I then used our gradient descent function to get the optimal \\(\theta\\) values on the training set. I then predict values for the testing set. 

```python
theta_init_reg = np.zeros((len(X_train_norm[0]),))
theta_learned_reg, _ = gradient_descent(
    X_train_norm, y_train, theta_init_reg, 0.1, 1000, 1.0
)
y_pred_reg = predict(X_test_norm, theta_learned_reg)
```

Since we used L2 normalization here, I will compare my method to the `Ridge` regressor from `scikit-learn`, using the same regularization strength of 1: 

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1)
ridge.fit(X_train_norm[:, 1:], y_train)
preds_ridge = ridge.predict(X_test_norm[:, 1:])
```

when comparing RMSE values computed on `y_pred_reg` and `preds_ridge`, we get almost identical RMSE values *(which hopefully makes sense to you)*. 

```plaintext
Our regularized model RMSE on test set: 4.971474178571149
Scikit-model ridge RMSE on test set: 4.971527511444203
```

# Parting words

All in all linear regression is super simple to implement, I hope you have a better understanding of how it works and why regularization is included in linbraries like `scikit-learn`. By the way the concept of regularization as a way to fight overfitting is very common in machine learning even in other methods than linear regression.  
I hope you had fun reading this short series and if you have any feedback do not hesitate to let me know.