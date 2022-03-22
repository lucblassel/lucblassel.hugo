---
title: Implementing linear regression, math and Python!
date: 2020-01-09 10:00:00
excerpt_separator: <!--more-->
feature: https://upload.wikimedia.org/wikipedia/commons/e/ed/Residuals_for_Linear_Regression_Fit.png
latex: true
tags:
  - code
  - machine-learning
  - python
  - regression
  - supervized
  - linear-model
  - algorithmic
---

Today I want to explain linear regression. It is one of the simplest statistical learning models and can be implemented in only a couple lines of Python code, in an efficient manner. Being so simple however does not mean it is not useful, in fact it can be very practical to explore relationships between features in a dataset and make predictions on a target value. Therefore I think it's important to understand how the method works and how the different parameters have an effect on the outcome.

<!--more-->

# What is linear regression ?

Like it's name implies it is a regression method, therefore it learns a relationship between input features and a continuous numerical value, for example, given the number of rooms in an apartment and the crime rate in its neighborhood, guess the value of that apartment. The _linear_ means that the inferred relationship between features and target values is linear: the target value is equal to the sum of the feature values multiplied by coefficients. What the method learns is those coefficients in order to fit to the training data as closely as possible.

## What does it mean in practice ?

Ok, let's start. Let's assume, for simplicity's sake, that we have only one feature \\(x\\) associated to a target value \\(y\\) that is equal to \\(2\cdot x\\) with noise added. If we plot this dataset we get this figure, with the red dotted line corresponding to the line of equation \\(y = 2\cdot x\\).

![example dataset](example_dataset_linear.svg)

The linear regression will make the assumption that the underlying equation of this dataset is:

$$
y = \theta\cdot x
$$

and if the algorithm is implemented correctly it will return \\(\theta \approx 2\\).

This is all well and good but what if the equation that generated the dataset is no longer \\(y = 2\cdot x\\) but is affine and \\(y = 2\cdot x + 3\\)?  
In this case our model cannot get a good fit to the data if it assumes that the equation is of the form \\(y = \theta\cdot x\\). To take this into account we usually add a bias term to the features that is always equal to \\(1\\) and learn 2 parameters \\(\theta_0\\) for the bias and \\(\theta_1\\) for the \\(x\\) feature, this way the regression fits the following equation:

$$
y = \theta_0\cdot 1 + \theta_1\cdot x
$$

and if the methods works well we should find \\(\theta_0\approx 3\\) and \\(\theta_1\approx 2\\).

This works as well if we have several features \\(x_1\\) and \\(x_2\\) for example, in this case we fit the equation \\(y = \theta_0\cdot 1 + \theta_1\cdot x_1 + \theta_2\cdot x_2\\), and so on and so forth.  
The general notation for this if we have \\(n\\) features \\(x_1, x_2, \cdots, x_n\\) becomes:

$$
y = \theta_0 + \sum_{i=1}^{n} \theta_i\cdot x_i
$$

## How do we compute \\(\theta\\) values ? 

Ok so first let's agree on some notation, we have:

- \\(X\\) the set of features for every training example, it is a table _(ie. matrix)_ where each row is an example and each column is a feature. The dataset presented in above has 15 points and only one feature \\(x\\) so in this case \\(X\\) has 15 rows and only 1 column.
- \\(y\\) is a column _(ie. vector)_ containing the target value for each training example.
- \\(x^{(i)}\\) is the \\(i_{th}\\) training example (ie. the \\(i^{th}\\) row of \\(X\\)), alternatively \\(y^{(i)}\\) is the \\(i^{th}\\) target value.
- \\(x_{j}\\) represents the \\(j^{th}\\) feature, (ie. the \\(j^{th}\\) column of \\(X\\) ).
- \\(\theta_i\\) is the coefficient associated to \\(x_i\\)
- \\(\Theta\\) is a list _(ie. row vector)_ containing all the coefficients \\(\theta_i\\) for each feature \\(x_i\\).
- \\(\hat{y}\\) is the column _(ie. vector)_ of target values predicted by the linear regression model, and \\(\hat{y}^{(i)}\\) its \\(i^{th}\\) value.
- \\(m\\) the number of training examples and \\(n\\) the number of features.

The way linear regression works is by finding the coefficients that result in the minimal distance between each target value and its corresponding prediction, this distance is represented in the following figure as green lines. As we can see the distance between the prediction _(dotted line)_ and real values _(blue points)_ is much lower when \\(\theta = 2\\) _(which is the coefficient that generated the data)_ on the left than when \\(\theta = 6\\) on the right.

![distance to fitted line](square_distance.svg)

In actuality what we want to minimize is the average of the sum of square distances, which will be our cost function \\(C\\), which we can write this way:

{{<longmath>}}
$$
\begin{aligned} C &= \frac{1}{2m}\sum^{m}_{i=1}(\hat{y}^{(i)} - y^{(i)})^2 \\\\
    with\ \ \hat{y}^{(i)} &= \theta_0 + \sum_{j=1}^n \theta_j\cdot x^{(i)}_j \end{aligned}
$$
{{</longmath>}}

How do we find the minimal value for \\(C\\), the first thing to note here is that the only values we have any control over are the \\(\hat{y}^{(i)}\\), and within that the only values we can change are the \\(\theta\\) values, so we change these coefficients until we find the minimal value of \\(C\\).  
Let's take our first dataset again where the target values \\(y\\) are roughly twice the single feature \\(x\\), without any bias. Therefore our linear regression algorithm only has to find one \\(\theta\\). We can compute the cost for several different values of \\(\theta\\) and see which one has the best cost which is show in the next figure:  
![cost in function on theta](cost_gradient_simple.svg)  
Here we can see that the cost is minimal when \\(\theta=2\\) which is what we want. However it is not very practical, nor efficient, to compute the cost for many values of \\(\theta\\), especially once we have more than one feature. So we need a way to avoid having to compute all costs and only the ones that will help us guess the minimum correctly, to do this we use _gradient descent_.

For those that have done some algebra before, you'll know everything about derivatives and gradients, for the others I'll do a very quick recap. The derivative of a function \\(f(x)\\) with regards to \\(x\\), \\(\frac{df}{\delta x}\\), describes the rate of change of \\(f\\). So when \\(\frac{df}{\delta x}\\) is positive, it means \\(f\\) is growing when \\(x\\) grows and when \\(\frac{df}{\delta x}\\) is negative, \\(f\\) is decreasing when \\(x\\) grows.  
If you have a function that depends on several variables, like \\(g(x_1, x_2)\\) we can compute several partial derivatives, each with regards to one of the variables of our function. In our case these partial derivatives would be: \\(\frac{\delta g}{\delta x_1}\\) and \\(\frac{\delta g}{\delta x_2}\\). The gradient is simply a vector containing all the partial derivatives of our function.

If we compute the gradient of our cost function \\(C\\) with regards to the different \\(\theta\\) coefficients, we can tell how we need to adjust a particular \\(\theta\\) value to lower the cost. Let's work out the partial derivative \\(\frac{\delta C}{\delta\theta_k}\\) for a given \\(k\\) _(This might be a little math intensive so if you're not interested you can just skip to the end of the derivation)_

{{<longmath>}}
$$
\begin{aligned}
    \frac{\delta C}{\delta\theta_k} &= \frac{\delta}{\delta\theta_k}(\frac{1}{2m}\sum_{i=1}^m(\hat{y}^{(i)} - y^{(i)})^2)\\
    &= \frac{1}{2m}\sum_{i=1}^m\frac{\delta}{\delta\theta_k}(\hat{y}^{(i)} - y^{(i)})^2\\
    &= \frac{1}{2m}\sum_{i=1}^m\frac{\delta}{\delta\theta_k}(\hat{y}^{(i)} - y^{(i)})\cdot2\cdot(\hat{y}^{(i)} - y^{(i)})\\
    &= \frac{1}{m}\sum_{i=1}^m\frac{\delta}{\delta\theta_k}(\hat{y}^{(i)} - y^{(i)})\cdot(\hat{y}^{(i)} - y^{(i)})
\end{aligned}
$$
{{</longmath>}}

We can then work out \\(\frac{\delta}{\delta\theta_k}(\hat{y}^{(i)} - y^{(i)})\\):

{{<longmath>}}
$$
\begin{aligned}
    \frac{\delta}{\delta\theta_k}(\hat{y}^{(i)} - y^{(i)}) &= \frac{\delta}{\delta\theta_k}(\sum_{j=0}^n\theta_jx^{(i)}_j - y^{(i)})\\
    &= \frac{\delta}{\delta\theta_k}(\theta_kx^{(i)}_k + \sum_{j\neq k}\theta_jx^{(i)}_j - y^{(i)})\\
    &= \frac{\delta}{\delta\theta_k}(\theta_kx^{(i)}_k) = x^{(i)}_k
\end{aligned}
$$
{{</longmath>}}

So finally we can have our partial derivative:

{{<longmath>}}
$$
\frac{\delta C}{\delta\theta_k} = \frac{1}{m}\sum_{i=1}^m(\hat{y}^{(i)} - y^{(i)})\cdot x^{(i)}_k
$$
{{</longmath>}}

### How do we use the gradient ?

Let's assume we have our gradient for \\(C\\) composed of all the \\(\frac{\delta C}{\delta \theta_j}\\). Let's imagine then that \\(\frac{\delta C}{\delta \theta_1} > 0\\), this means that our cost is increasing with \\(\theta_1\\), meaning that if we want to decrease \\(C\\) we must also decrease \\(\theta_1\\) by a certain amount. By doing this for all the \\(\theta_j\\) we can decrease \\(C\\) and if we repeat this step a large enough number of times we can converge to the minimal value of \\(C\\). We know that when we reach the minimum of \\(C\\), then the gradient value should all be equal to 0.

If we try to summarize the steps that we will have to implement for linear regression we have:

1.  compute the cost \\(C\\) with all the \\(\theta_j\\) values
2.  compute the gradient of \\(C\\) with respect to all the \\(\theta_j\\)
3.  adjust all \\(\theta_j\\) values according to the corresponding partial derivative: if \\(\frac{\delta C}{\delta \theta_j} < 0\\) increase \\(\theta_j\\) a little otherwise decrease it.

We then repeat these steps with the adjusted values for all the \\(\theta_j\\) until we have a gradient that is equal to 0, or more commonly if we reach a predefined maximum number of iterations.

And that's it, this is the principle behind linear regression. However before starting the implementation there are a few things I need to explain still.

### How do we adjust \\(\theta\\) ?

Just above I mentioned that if \\(\frac{\delta C}{\delta \theta_j} < 0\\) we need to increase \\(\theta_j\\), but by how much do we increase it?  
The basic idea is that we subtract the partial derivative value from the current \\(\theta_j\\) value, therefore if the partial derivative is negative by subtracting a negative to \\(\theta_j\\) we increase it like we must. However in practice we multiply the partial derivative by a learning rate \\(\alpha\\) that we must choose before subtracting it. This learning rate will allow us to adjust how fast our regression "learns", it's choice is very important, if \\(\alpha\\) is too small we will barely change the \\(\theta\\) values and our regression will find the minimum very slowly, and inversely if \\(\alpha\\) is too big our cost will jump around everywhere and we might not find the minimum at all and our cost might actually increase as we adjust the \\(\theta\\) values. I'll come back to this later.

### Can we write the math in another way?

In this part I will show you how to write all the math in matrix form, because this will make the programming part a lot easier and more efficient to run _(This means that you have to know the matrix math basics to understand this)_.  
As we said earlier we have our training examples \\(X\\), where each example \\(x^{(i)}\\) is a row and where each feature \\(x_j\\) is a column. Therefore if we have \\(m\\) examples and \\(n\\) features, then \\(X\\) is an \\(m\times n\\) matrix. We also have our vector of target values \\(y\\) which is an \\(m\times 1\\) vector. If you recall earlier we had our predicted value for \\(x^{(i)}\\):

$$
\hat{y}^{(i)} = \theta_0 + \sum_{j=1}^{n}\theta_j\cdot x^{(i)}_j
$$

if we add a feature \\(x_0 = 1\\) to our input vector, our prediction becomes:

$$
\hat{y}^{(i)} = \sum_{j=0}^{n}\theta_j\cdot x^{(i)}_j
$$

So now we have our input vector \\(x^{(i)}\\) of dimension \\(1\times n+1\\) and our coefficient vector \\(\Theta\\) of dimension \\(n+1\times1\\), so computing \\(\hat{y}^{(i)}\\) can be done with a single vector multiplication, and computing the prediction vector \\(\hat{y}\\) containing all predictions can be done in a single matrix multiplication.

{{<longmath>}}
$$
\begin{aligned}
    \hat{y}^{(i)} &= x^{(i)}\cdot\Theta\\
    \hat{y} &= X\cdot\Theta
\end{aligned}
$$
{{</longmath>}}

We can therefore write our cost as follows, where \\(X^{\circ2}\\) means we apply the square function to each element of the matrix \\(X\\) separately:

{{<longmath>}}
$$
\begin{aligned}
    C &= \frac{1}{2m}\sum_{i=1}^m(\hat{y}^{(i)} - y^{(i)})^2\\
    &= \frac{1}{2m}\sum_{i=1}^m(x^{(i)}\cdot\Theta - y^{(i)})^2\\
    &= \frac{1}{2m}\sum((X\cdot\Theta - y)^{\circ2})
\end{aligned}
$$
{{</longmath>}}

We can also write the gradient \\(\nabla C\\) in matrix math form:

{{<longmath>}}
$$
\nabla C = \begin{bmatrix}
        \frac{\delta C}{\delta\theta_0}\\
        \frac{\delta C}{\delta\theta_1}\\
        \vdots \\
        \frac{\delta C}{\delta\theta_n}
    \end{bmatrix}
$$
{{</longmath>}}

Where each partial derivative can be computed as follows:

{{<longmath>}}
$$
\begin{aligned}
    \frac{\delta C}{\delta\theta_k} &= \frac{1}{m}\sum_{i=1}^m(\hat{y}^{(i)} - y^{(i)})\cdot x^{(i)}_k \\
    &= \frac{1}{m}x_k^T\cdot(X\cdot\Theta - y)\\
\end{aligned}
$$
{{</longmath>}}

Which gives us:

$$
\nabla C = \frac{1}{m}X^T\cdot(X\cdot\Theta - y)
$$

This matrix form will allow us to write the code more clearly once we get to that, and the computations will be much more efficient with matrix math than writing loops for the sums.  
Ok so I guess we can start getting into the implementation part of this post. I'll do this in Python but any other programming language with a decent matrix math library will do the trick.

# How can we implement linear regression ?

## Getting a dataset

Ok so first things first we need some data on which to train our linear regressor, I'm going to stick to basics an use the <a></a>[boston housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html), where we try to guess the median monetary value of different homes depending on several features like the number of rooms, crime rate, distance to the nearest job center, etc...  
This dataset is available in the `scikit-learn` library in Python and we are going to split it into a training dataset with \\(80\%\\) of the examples and keep the remaining \\(20\%\\) as a testing set on which we can evaluate the performance of our linear regressor. I wrote a small dataset splitting function, and loaded the data.

```python
import numpy as np
from sklearn.datasets import load_boston

def split_dataset(X, y, train_frac=0.8):
    index = np.random.choice(len(y), int(len(y) * train_frac))
    X_train, y_train = X[index], y[index]
    X_test, y_test = np.delete(X, index, 0), np.delete(y, index, 0)
    return X_train, y_train, X_test, y_test

X, y = load_boston(return_X_y=True)
X_train, y_train, X_test, y_test = split_dataset(X, y)
```

Our dataset looks like this:

{{<markdowntable>}}
| CRIM    | ZN   | INDUS | CHAS | NOX   | RM    | AGE  | DIS    | RAD | TAX   | PTRATIO | B      | LSTAT |
|---------|------|-------|------|-------|-------|------|--------|-----|-------|---------|--------|-------|
| 0.00632 | 18.0 | 2.31  | 0.0  | 0.538 | 6.575 | 65.2 | 4.0900 | 1.0 | 296.0 | 15.3    | 396.90 | 4.98  |
| 0.02731 | 0.0  | 7.07  | 0.0  | 0.469 | 6.421 | 78.9 | 4.9671 | 2.0 | 242.0 | 17.8    | 396.90 | 9.14  |
| 0.02729 | 0.0  | 7.07  | 0.0  | 0.469 | 7.185 | 61.1 | 4.9671 | 2.0 | 242.0 | 17.8    | 392.83 | 4.03  |
| 0.03237 | 0.0  | 2.18  | 0.0  | 0.458 | 6.998 | 45.8 | 6.0622 | 3.0 | 222.0 | 18.7    | 394.63 | 2.94  |
| 0.06905 | 0.0  | 2.18  | 0.0  | 0.458 | 7.147 | 54.2 | 6.0622 | 3.0 | 222.0 | 18.7    | 396.90 | 5.33  |
| ...     | ...  | ...   | ...  | ...   | ...   | ...  | ...    | ... | ...   | ...     | ...    | ...   |
{{</markdowntable>}}

The next step we need to do is normalize the features, because as we can see above they are not at all in the same scale, for example `CRIM` is a lot smaller than `TAX` for example so in our cost function `TAX` will have a much larger impact than `CRIM`. To mitigate that we bring all the features to the same scale by subtracting the mean and then dividing by the standard deviation.

$$
x_j^{normalized} = \frac{x_j - \mu(x_j)}{\sigma(x_j)}
$$

The effect of normalization on the value distribution of each feature is shown after. On the left the features are not normalized and all other features are dwarfed by the `TAX` and `B` features, whereas on the right, after normalizing all the features have roughly the same scale and therefore the same impact on cost.

![feature normalization effect](feature_normalization.svg)

We compute the \\(\mu\\) and \\(\sigma\\) values for each feature only on the training set and use these values to normalize the testing set as well. I wrote a little function to help us do that:

```python
def normalize(X, mu=None, sigma=None):
    if mu is None or sigma is None:
        mu, sigma = X.mean(axis=0), X.std(axis=0)
    return ((X - mu) / sigma), mu, sigma

# feature normalize
X_train_norm, mu, sigma = normalize(X_train)
X_test_norm, _, _ = normalize(X_test, mu, sigma)
```

Before we can get to training our regressor we still need the bias feature \\(x_0\\) to our \\(X\\) matrix. To do that we just add a column full of ones to the beginning of \\(X\\):

```python
# add bias feature
X_train_norm = np.append(
  np.ones((len(X_train_norm), 1)), X_train_norm, axis=1)
X_test_norm = np.append(
  np.ones((len(X_test_norm), 1)), X_test_norm, axis=1)
```

## Building our regressor

Now we can get to the actual regression part. First we need to be able to compute our cost, note that in Python the matrix multiplication symbol is `@`, and the `**2` means we square the matrix element-wise.

```python
def compute_cost(theta, X, y):
    return sum((X @ theta - y) ** 2) / (2 * len(y))
```

In the same fashion we can also compute the gradient:

```python
def compute_gradient(theta, X, y):
    return (X.T @ (X @ theta - y)) / len(y)
```

And that's basically it, we have all the elements needed for our gradient descent. To keep it simple I'm going to end the gradient descent after a set number of iterations `max_iters` but you could as well add a stopping condition if the cost doesn't change. The way we do this is at each iteration we update the \\(\theta\\) coefficients by subtracting the gradient from the \\(\Theta\\) vector weighted by the learning rate \\(\alpha\\). To have an idea of what the linear regression is doing during training we keep the list of costs in the `history` list to be able to plot the learning curve, this curve can be quite useful to check that our regressor is behaving as expected and that we have the correct learning rate.

```python
def gradient_descent(X, y, theta, alpha, max_iters):
    history = []
    for _ in range(max_iters):
        theta = theta - alpha * compute_gradient(theta, X, y)
        history.append(compute_cost(theta, X, y))
    return theta, history
```

## How do we choose the learning rate ?

Here I am just going to try a couple different \\(\alpha\\) values and check the learning curves for each of these alpha values. With small values, like \\(\alpha=0.001\\) you can see that the cost decreases so we are converging but it takes a very long time, but as \\(\alpha\\) gets bigger the cost decreases faster and faster.

![learning rate effect](learning_rate.svg)

But we have to be careful, if we choose an \\(\alpha\\) value that's too big it the cost can end up growing instead of decreasing and we do not find the optimal value for \\(\Theta\\):

![learning rate too big](learning_rate_too_big.svg)

So after looking at these graphs we can choose the right value \\(\alpha = 0.1\\) which will give us the fastest convergence time.

## training our regressor

So now we can actually train our linear regression and then measure it's performance on our test data set. First we need initial \\(\theta\\) values, so we are just going to set them all equal to zero.

```python
# Initializing Theta vector
theta_init = np.zeros((len(X_train_norm[0]),))
# learning optimal Theta vector
theta_learned, cost_history = gradient_descent(
    X_train_norm, y_train, theta_init, 0.1, 1000)
```

Now that we have our \\(\Theta\\) vector we can use it to make predictions on the normalized test set. We can then check how well our model performs by plotting the predicted values `y_pred` against the real values `y_test`.

```python
# To get y^ predicted values
def predict(X, theta):
    return X @ theta

y_pred = predict(X_test_norm, theta_learned)
```

![regression plot](regression_plot.svg)

The red dotted line shows the \\(y = x\\) line, so if our regressor was perfect all the points would be exactly on that line, however nothing is perfect. All the points are close to that diagonal line, meaning our linear regressor is actually doing quite well, except for high true values where the predictions are consistently lower than the truth.
One last thing we can do is compare our method to the state of the art implementation from the `scikit-learn` library.

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
# We remove the bias feature as it is automatically added by scikit-learn
reg.fit(X_train_norm[:, 1:], y_train)
preds_sklearn = reg.predict(X_test_norm[:, 1:])
```

TO compare the performance of our model and the performance of the state of the art model, we need a metric like the root mean square error (RMSE):

$$
RMSE = \sqrt{\frac{\sum_{i=1}^m(\hat{y}^{(i)} - y^{(i)})^2}{m}}
$$

In this case our regressor and the `scikit-learn` regressor both have an RMSE of \\(4.9277\\), with only a difference of \\(0.00008\%\\), so our model and the state of the art have an identical performance, which is quite reassuring.

In the [next post]( {{<relref "posts/regularization" >}} ) I'll talk about regularization and how we can add it to our linear regressor. In the meantime I hope you've liked this little write-up and if you want to take a closer look at the code it's all available in this repo: <a></a>[github.com/lucblassel/website_projects](https://github.com/lucblassel/website_projects/)
