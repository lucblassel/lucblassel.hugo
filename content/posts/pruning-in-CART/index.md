---
title: How does pruning work in CART ?
excerpt_separator: <!--more-->
draft: true
latex: true
feature: https://images.unsplash.com/photo-1515879218367-8466d910aaa4?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80
latex: true
tags:
  - decision-trees
  - machine-learning
  - classification
  - regression
  - supervized
  - code
  - python
  - algorithmic
---

Ok so as we saw in previous parts, the CART algorithm allows us to build decision trees. Up till now we have built these trees until all leaves are pure, meaning they have only one class of examples (for classification trees), however this can lead to overfitting the training data which decreases the generalizability of our model, and therefore it's usefulness. This is where cost-complexity pruning comes into play.

<!--more-->

# What is pruning?

So [pruning](https://en.wikipedia.org/wiki/Pruning) comes from biology, pruning a plant is selectively removing some part of it. In the case of decision trees, it just means removing some branches. However even though we remove branches we want to keep all of our samples, so we cant just eliminate part of the samples from branches, so effectively removing a branch corresponds to choosing a pruning node, where we want our branch to end, and collapsing all it's child nodes into it.  
Now how do we choose which branches to remove ? if we remove too many our model looses any classifying, or regressing power it has and if we remove too few we can still have overfitting. This is addressed by cost-complexity pruning, which balances the complexity of the tree _(the number of leaves, so potential overfitting)_ with the performance of the tree.

# Notation

In order to explain cost-complexity pruning, we are going to need to give some names to things we need, luckily that's already been done.

## Tree nomenclature

Let us consider a decision tree $T$ and two of its nodes, $t$ and $t'$.

- $t'$ is a **descendant** of $t$ if there is a path **down** _(from the root to the leaves)_ the tree from $t$ to $t'$.
- $t$ is an **ancestor** of $t'$ if there is a path **up** _(from the leaves to the root)_ from $t'$ to $t$.
- $t_R$ and $t_L$ are, respectively, the right and left child nodes of $t$
- A **branch** $T_t$ is the branch of $T$ with root $t$, is composed of the node $t$ and all of its descendants.
- **pruning** a branch $T_t$ from $T$ is removing all nodes of $T_t$ from $T$, the **pruned tree** is called $T-T_t$
- If you can get a tree $T'$ from $T$ by pruning branches, the $T'$ is a **pruned subtree** of $T$ and we denote that with: $ T' \leq T$
- For a given tree $T$, we can define, the **set of leaf nodes** $\widetilde{T}$
- The **complexity** of $T$ is given by the cardinality of $\widetilde{T}$, _(ie. the number of leaf nodes)_, it is noted: $\vert\widetilde{T}\vert$

## Measures

Let us consider a leaf node $t$ of $T$, with $\kappa(t)$ the class of $t$ _(ie the majority class in the node)_.

- $r(t) = 1 - p(\kappa(t)\vert t)$ the is the **resubstitution error estimate** of $t$. $p(\kappa(t)\vert t)$ is the proportion of the majority class in $t$.
- We denote $R(t) = p(t)\cdot r(t)$, with $p(t)$ simply being the proportion of samples in node $t$ compared to the rest of the tree.
- It is provable that $R(t) \geq R(t_R) + R(t_L)$, which just means that if we split a node the misclassification rate is sure to improve.
- The overall **missclassification rate** for $T$, is:

$$
R(T) = \sum_{t\in \widetilde{T}} R(t) = \sum_{t\in \widetilde{T}} r(t)\cdot p(t)
$$

Which is to say the sum of the resubstitution error of a leaf node multiplied by the probability of being in said node over all of the leaf nodes.

# The pruning

The first step in pruning a tree is, ..., you guessed it: having a tree. So we start by growing $T_{max}$ the maximal tree, with pure leaves. Now the naive approach would be to go through all the possible pruned subtrees and see which one has the best trade-off between performance and complexity, however that is, in practice, impossible because of the huge number of possible pruned subtrees.

This is a test of how the **_git_** thing works
