---
title: The CART Algorithm
date: 2019-02-27 10:00:00
excerpt_separator: <!--more-->
tags: [decision-trees, machine-learning]
feature: https://images.unsplash.com/photo-1509228468518-180dd4864904?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80
latex: true
---

This is Part 2. of my decision tree series. Here we will see how we can build a decision tree algorithmically using Leo Breiman's (One of the big, **big** names in decision trees) CART algorithm.

## <!--more-->

## Maths (yay!)

Ok, so there's going to be a little maths in this post, so for those who are not sure what all the notations mean let's go over them. (if you already know all this feel free to skip ahead 😄)

### Set notation

Anything in between brackets $\{\}$ is a set of objects (generally numbers).

- $\{1,3,4\}$ is the set that is composed of $1$, $3$ and $4$.
- $\{1,\cdots, 7\}$ is the set of all integers between $1$ and $7$
- $i \in \{1,3,4\}$ means $i$ is equal to either $1$, $3$ or $4$ (it is read $i$ **in** $\{1,3,4\}$)
- $i \notin \{1,3,4\}$ means $i$ is any number tht is not $1$, $3$ or $4$ (it is read $i$ **not in** $\{1,3,4\}$)
- $\{x \leq 4,\ x\in\{1,\cdots,6\}\}$ means the set of $x$ that less or equal to $4$ and $x$ in $\{1,\cdots,6\}$, _(here this means $x\in\{1,2,3,4\}$)_
- $A\subset B$ means that the set $A$ a subset of the set $B$ and it cannot be equal to $B$. For example $\{1,2\} \subset \{1,\cdots,8\}$ is **true** since $1$ and $2$ are included in the set of integers between $1$ and $8$, but $\{1,2\} \subset \{2,\cdots,8\}$ is **not true** since $1$ is not included in the set of integers between $2$ and $8$ _(we could say $\{1,2\}\not\subset \{2,\cdots,8\}$)_.
- If we have a set $\mathcal{A}$ then $\overline{\mathcal{A}}$ is it's opposite set. So if $\mathcal{A}=\{1,2,3\}$ then $\overline{\mathcal{A}} = \{x \notin \{1,2,3\}\}$

### Sums

Sums are represented by the $\Sigma$ symbol _(read as "sigma")_. The value under the sigma is the start point, and the value above is the end point of the sum. So :

$$
\sum^3_{i=1} i^2 = 1^2 + 2^2 + 3^2 = 14
$$

If you have any more questions about mathematical notation I would suggest you consult <a></a>[this Wikipedia page][1] which is quite comprehensive.

### in this article

- $Y$ denotes the target variable _(in our <a></a>[iris dataset][2] it would be the species)_
- $X_{\{1,\cdots,p\}}$ are the $p$ explanatory variables _(in the iris dataset: petal length and width, and sepal length and width, so $\ p=4$)_

For a classification tree $Y_i \in \{1,2,\cdots,k\}$, where $k$ is the number of possible classes.  
On the other hand for a regression tree $Y_i \in \mathbb{R}$ _(with $\mathbb{R}$ the set of real numbers)_  
_(NB. in any post I might use "variable" and "feature" interchangeably because they essentially mean the same thing)_

In decision trees nodes represent a split in data, in this post I will usually call splits $S$ and represent them as the 2 opposite sets $\mathcal{A}$ and $\overline{\mathcal{A}}$, where $\mathcal{A}$ is the set that goes to the left branch of the split and $\overline{\mathcal{A}}$ the set that goes to the right branch of the split. For example, the first split of <a></a>[part 1][2]'s simple tree, that has the condition $x_3 \leq 1.9$ will be noted as:

$$
S = \{x_3 \leq 1.9\},\ \{x_3 > 1.9\}
$$

<br/>
<br/>
<br/>

## Steps in the CART decision tree inference algorithm

## What is CART?

CART stands for **C**lassification **A**nd **R**egression **T**rees, it is an algorithm developed by <a></a>[Leo Breiman _et. al_ in 1984][3] designed for inferring decision trees. It relies on evaluating possible splits at each node in our tree and choosing the best one.

## splitting the data

To infer a tree we need to choose the best possible split at each node, and to do that we need to know what those splits are. As we saw in <a></a>[part 1][2] a split is only ever done on one feature  
There are two types of features, hence two types of splits:

- **numerical features**: these have number values (usually continuous) with no fixed set of values. These values intrinsically have an order.
- **categorical features**: these are features that can take one value from a fixed set. For example a "blood type" feature could only have values from the set $\{A,\ B,\ AB,\ O\}$, or a "group number" feature could have values only in set $\{1,\cdots,10\}$. It is important to know that categorical features can be numbers, they just have to by limited to a specific finite set of possible values. These values are unordered.

_(NB. there can be ordered categorical values, like grades for example $A > B > C > D$ but they can be assimilated to numerical values which is why I didn't mention them in the list above)_

#### A. numerical splits

For a given node there are $n$ data points. Let us consider the numerical explanatory feature $X_{num}$. In this case we have a maximum number of splits $n-1$ corresponding to all the splits:

$$
S = \{ X_{num} \leq x_i\},\{X_{num} > x_i\}\quad\quad i \in \{1,\cdots,n-1\}
$$

We stop at $n-1$ because, for the maximum value of $X_{num}$ all points are smaller or equal to it, meaning we send all of our data points to one side of the split and leave the other side empty. This of course is not split, so we don't count that possibility.

#### B. categorical splits

Let us consider now the categorical feature $X_{cat}$, which has $k$ possible levels. the possible number of splits is $2^{k-1} - 1$. A given split can be defined as:  
<a id='categorical-splits-definition'> <a/>

$$
S = \{X_{cat} \in \mathcal{A}\},\{X_{cat} \in \mathcal{\overline{A}}\}\quad\quad \mathcal{A}\subset\{1,\cdots,k\}
$$

_How many subsets $\mathcal{A}$ are there?_  
Let's consider the case of a feature called $C$ with 3 levels:

$$
C \in \{red,\ blue,\ green\}
$$

To create a subset we must choose which values are in or not the subset. For example the subset $\mathcal{A} = \{red,\ green\}$ what we are saying is $red\in \mathcal{A}$ and $green\in \mathcal{A}$ and $blue\notin \mathcal{A}$. So each subset is a set of 3 values indicating the presence or absence of a given level of $C$ in the subset. With this we can easily calculate the number of possible subsets.  
For the first value (presence or absence of $red$ in the subset) there are $2$ possible options, for the second value we also have $2$ potential values and the same for the third values.  
So our total number of possible subsets is:

$$
N_{sets}=2\times2\times2 = 2^3
$$

And if we have $k$ possible levels, our number of possible subsets becomes:

$$
N_{sets}=2^k
$$

_but further up it was_ $2^{k-1} -1$ _why?_  
Well this comes from the fact that we are looking for splits, not just for subsets. And splits are symmetrical. For our example above, if we have:

$$
\begin{aligned}
    \mathcal{A} &= \{red, green\} \Leftrightarrow \mathcal{\overline{A}} = \{blue\} \\
    \quad\\
    S_1 &= \{C\in\mathcal{A}\},\{C\in\mathcal{\overline{A}}\}\\
    S_2 &= \{C\in\mathcal{\overline{A}}\},\{C\in\mathcal{A}\}\\
\end{aligned}
$$

It is easy to see that $S_1 = S_2$, that symmetry is why on out number of possible splits we have $2^{k-1}$ instead of simply $2^k$. The explanation for why it is $2^{k-1}-1$ and not $2^{k-1}$ is the same as for numerical features, because if $\mathcal{A}=\{red,\ blue,\ green\}$ then Our splits has all the points one one side and and empty set on the other, so it is not a split, so we remove that possibility and that's how we end up with $2^{k-1}-1$ _(thats also why we used $\mathcal{A} \subset \{1,\cdots,k\}$ instead of $\mathcal{A} \subseteq \{1,\cdots,k\}$ when we_ <a></a>[defined our categorical splits](#categorical-splits-definition) _)._

So now we have all of the possible splits in our data, but how do we choose the best one?

## Choosing the best split

Since we want to use the tree to predict either a class or a value, we want the leaves of the tree to be as "pure" as possible, meaning we want the examples in each leaf to be similar. To get that we need a way to measure the "purity" of a node, so how similar all data points are in that node.  
Therefore, to chose the best split, we choose the one that maximizes this "purity" measure in the child nodes. In practice we don't measure "purity" but rather "impurity". There are several of these measures. In this whole section let's consider a node $t$, and for an impurity measure $i$ we can define $i(t)$ as the impurity of this node.

### The Gini index

The Gini index is a way to measure impurity in classification trees. Let $p_i$ be the probability of having class $i$ in our node, and $k$ the number of classes in the node. The Gini index $G$ is:

$$
G(t) = 1 - \sum^k_{i=1} p_i^2
$$

Since we don't know the real probability $p_i$ for a given class, we just approximate it by the frequency of said class in the node:

$$
p_i = \frac{n_i}{n}
$$

with $n_i$ the number of examples of class $i$ in the considered node and $n$ the total number of examples in said node.

### Entropy

Entropy is also an impurity measure that is commonly used in CART with classification trees. If we use the same notation as for the Gini index we can define the Entropy of a node, $E$, as:

$$
E(t) = \sum^k_{i=1} p_i \log(p_i)
$$

usually the base $2$ logarithm is used.

### RSS

The Residual Sum of Squares (RSS) is used in regression trees, where examples have an outcome value instead of an outcome class. For a given node, let $n$ be the number of examples at that node, $y_i$ the outcome value of the $i^{th}$ example, and $\mu$ the mean outcome values of all examples of the node. We can define :

$$
RSS(t) = \sum^n_{i=1} (y_i - \mu)^2
$$

<br/>  
<br/>  
<br/>

To then choose the optimal split we choose the one that leads to the maximal decrease in impurity _(which can be either the Gini index or the entropy for classification trees or the RSS for regression trees)_. So if we are in a node $t$ and a given split defines nodes $L$ and $R$ the **_left_** and **_right_** child nodes of $t$ respectively. We can define the decrease in impurity $\Delta i$ as:

$$
\Delta i = i(t) - p_L\cdot i(L) - p_R\cdot i(R)
$$

with $i(L)$ and $i(R)$ the impurities of the child nodes and $p_L$ and $p_R$ the probabilities of a sample from node $t$ will go to node $L$ or node $R$, again we equate this to the proportions of cases that go to nodes $R$ and $L$ so $p_L = \frac{n_L}{n_t}$ (with $n_L$ and $n_t$ the number of cases in nodes $L$ and $t$)

## The algorithm

The basic algorithm is actually quite simple:

1. Find all possible splits in the dataset
2. calculate the decrease in impurity for each of these splits
3. Choose split for which decrease in impurity is maximal
4. on each half of the split, start this process again

We can implement it in a recursive manner thusly:

```c
function infer_tree(dataset) {

    #stopping condition
    if dataset_is_pure(dataset){
        stop
    }

    splits = find_all_possible_splits(dataset)
    impurities = calculate_impurity_decrease(splits)

    # choose split with maximum decrease
    best_split = splits[max(impurities)]

    left_dataset = best_split.left
    right_dataset = best_split.right

    #recursive part
    infer_tree(left_dataset)
    infer_tree(right_dataset)

}
```

This is only a part of the algorithm, it results in a tree that grows until all leaves are pure (meaning that they contain examples of only one class), and the CART algorithm restricts that so we can have a more generalizable tree (no overfitting).

- you can define a minimum number of examples in each leaf, and if the dataset at a node is smaller or equal than that minimum then the node is not split and stays a leaf. (This is not typically used in CART)
- you can prune the tree and collapse leaves together, which we will see in a later part.

## Conclusion

I hope you learned something on how to build decision trees, and go check out <a></a>[part 3][4] to see a `Python` implementation of this algorithm.

[1]: https://en.wikipedia.org/wiki/List_of_mathematical_symbols
[2]: /blog/what-are-decision-trees
[3]: https://www.taylorfrancis.com/books/9781351460491
[4]: /blog/lets-implement-CART
