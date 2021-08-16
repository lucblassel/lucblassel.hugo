---
layout: post
title: "How can we make linear regression better? Regularization."
draft: true
excerpt_separator: <!--more-->
tags: [linear model, machine-learning, Python, code, regression]
feature: https://upload.wikimedia.org/wikipedia/commons/e/ed/Residuals_for_Linear_Regression_Fit.png
latex: true
---

If you haven't read my post on linear regression I invite you to do so [here](blog/2020-01-09-implementing-linear-regression), but basically it is a method for modelling the relationship between variables $$X_i$$ and a target feature $$y$$ in a linear model. This modelling is done through learning weights $$\theta_i$$ for each $$X_i$$ supposing that our model looks something like this:

$$
y = \sum_{i=1}^n\theta_i\cdot X_i
$$

<!--more-->

# What is regularization ?

![Regularization](/images/regularization/Regularization.svg)

# How do we implement it in Python
