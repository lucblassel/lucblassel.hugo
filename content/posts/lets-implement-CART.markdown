---
title: Let's implement the CART Algorithm
date: 2019-03-02 10:00:00
excerpt_separator: <!--more-->
tags: [decision-trees, machine-learning, Python, code]
feature: https://images.unsplash.com/photo-1515879218367-8466d910aaa4?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80
latex: true
---

This is Part 3 of my decision trees series. This time around we are going to code a decision tree in Python. So I'm going to try to make this code as understandable as possible, but if you are not familiar with <a></a>[Object Oriented Programming (OOP)](https://en.wikipedia.org/wiki/Object-oriented_programming) or <a></a>[recursion](<https://en.wikipedia.org/wiki/Recursion_(computer_science)>) you might have a tougher time.

<!--more-->

To make data handling easier, we are going to be using the wonderful `pandas` package, so if you don't know how to use it I highly recommend you learn to by reading <a></a>[their intro to pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html). _(there are also very good tutorials by <a></a>[Chris Albon](https://chrisalbon.com/) but they are more focused on a specific feature)_, but I'll just quickly go over the most important points. In `pandas` tabular data is stored in a `DataFrame` and these objects allow us to have named columns and rows and to easily subset data with these names. Ok so let's create a `DataFrame` from a subset of our iris data.

```python
>>> import pandas as pd
>>> iris_df = pd.DataFrame(iris_data, columns = column_names)
>>> iris_df
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) species
0                5.1               3.5                1.4               0.2  setosa
1                4.9               3.0                1.4               0.2  setosa
2                4.7               3.2                1.3               0.2  setosa
3                4.6               3.1                1.5               0.2  setosa
4                5.0               3.6                1.4               0.2  setosa
```

Ok so we have our `DataFrame`, and now we can select just the species column for example with:

```python
>>> iris_df['species']
0    setosa
1    setosa
2    setosa
3    setosa
4    setosa
Name: species, dtype: object
```

Or we can get the samples which have $$sepal\ width \leq 3.1$$:

```python
>>> iris_df[iris_df['sepal width (cm)'] <= 3.1]
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) species
1                4.9               3.0                1.4               0.2  setosa
3                4.6               3.1                1.5               0.2  setosa
```

So it is easy to see how this makes understanding what the code is doing easier and that's why I'm using `DataFrame`s instead of arrays and indices. This is however far from all `pandas` can do _(duh...)_, and it's just the basics of the basics.

# Representing a tree.

So what is a tree?  
A tree is a set of nodes, so we need to define a `Node` object which will store some properties about the node, as well as it's left and right children nodes.

What does a node of our decision tree need?  
A pandas DataFrame to represent the training data at this node, the target feature _(in the case of our iris example that would be the "species" feature)_. We need to have a left and right child as well, and if the node is a leaf node it needs to be able to give a prediction. This gives us the following `Node` class definition, where we feed the dataframe as well as the target feature upon initialization.

```python
class Node:

    def __init__(self, data, target):
        self.data = data # Dataframe with features and target
        self.target = target # name of target
        self.left = None
        self.right = None
        self.prediction = None
```

<br/>

Ok so now we have our `Node` class, how do we represent a tree? where do we keep all the nodes we will need. Well actually we do not need to create a `Tree` class:  
since each node stores its left and right children, we can access any and all nodes of the tree from the root node. So the whole tree can just be represented by the root node. For example if we want to get the _versicolor_ node in our simple tree we can access it from the root node with: `root.right.left`
![small tree](/images/simple_tree.svg)

# getting the splits

Ok, so we can just concentrate ourselves on the `Node` class. So if you remember how the algorithm works _(if not <a></a>[here is the post](/blog/the-CART-algorithm) where I explain it)_, we are going to need a way to find splits in our data. As we said in previous parts, there are categorical and numerical splits, so we need a way to determine if a feature is categorical or numerical, fortunately `pandas` has us covered, and we can make a simple function:

```python
from pandas.api.types import is_categorical, is_string_dtype, is_bool

def check_categorical(data):
    return is_categorical(data) | is_string_dtype(data) | is_bool(data)
```

So now that we can distinguish the two we can write this function that gets all possible splits in our dataset, and returns them as a dictionary

```python
def get_splits(self):
    features = self.data.columns.drop(self.outcome)
    all_splits = {}

    for feature in features:

        if check_categorical(self.data[feature]):
            all_splits.update(self.get_categorical_splits(feature))
        else:
            all_splits.update(self.get_numerical_splits(feature))

    return all_splits
```

As you can see gets all the features in our dataset (except the outcome), loops over them and checks if the feature is categorical or numerical. Depending on th feature type it calls a given splitting method. So how do we get our categorical and numerical splits? For the numerical splits we have this function :

```python
def get_numerical_splits(self, feature):
    splits = {}
    uniques = self.data[feature].unique()
    for value in uniques:
        if value != max(uniques):
            splits[(feature, value, 'numerical')] = self.data[self.data[feature] <= value]
    return splits
```

This returns all possible numerical splits in a dictionary where the key is a `tuple` of the feature name, the value on which the split is done and the type of split, and as a value the data that goes to the left side of the split _(the data that respects the split condition)_.  
For categorical features we are not going to follow exactly what I said in <a></a>[part 2](/blog/the-CART-algorithm), indeed the total number of splits is: $$2^{k-1} - 1$$, with $$k$$ the possible values of our feature, this can get huge very quickly. For example, a categorical feature with 25 levels (25 brands of car, or 25 different languages, whatever...), which can be easily attained in some datasets, would result in $$33554432$$ splits to evaluate, and that's just for one feature in one node. So this can get out of hand very quickly and slow our program to a crawl. So I'm going to make an executive decision here and say we will only consider splits made by a single level, for example `brand = Ford` and eliminate all splits made by combinations of levels: `(brand = Ford) or (brand = Chevrolet)`. This brings us back to a nice $$k$$ possible splits. So we can add this method to get categorical splits:

```python
def get_categorical_splits(self, feature):
    splits = {}
    for unique in self.data[feature].unique():
        splits[(feature, unique, 'categorical')] = self.data[self.data[feature] == unique]
    return splits
```

Ok all done, so now at a single node we can get all the dataset splits we want to evaluate by calling the `get_splits()` method.

# Computing impurity

Next we need a way to calculate the impurity of a split, in our case is going to be the Gini index. For reminder the Gini index $$G$$ for a node $$t$$ is defined as:

$$
G(t) = 1 - \sum^k_{i=1} p_i^2
$$

Where $$p_i$$ is the proportion of samples of class $$i$$ in the node data, and $$k$$ is the number of different classes. So let's add a method to our `Node` class to do just that:

```python
def get_gini(self, data):
    proportions = data[self.outcome].value_counts(normalize=True)
    return 1 - (proportions ** 2).sum() # the ** applies to all elements of the column
```

`value_counts()` is a `pandas` method that gets all unique values in a given column and returns their counts, the `normalize` option makes it return proportions instead of counts. The next step is computing the decrease in impurity $$\Delta i$$ (see part 2 for formula).

```python
def get_delta_i(self, subset):
    gini = self.get_gini(self.data)

    left = subset
    right = self.data.drop(subset.index, axis=0)

    p_left = len(left) / len(self.data)
    p_right = 1 - p_left

    sub_left = p_left * self.get_gini(left)
    sub_right = p_right * self.get_gini(right)

    return gini - sub_left - sub_right
```

Here `subset` is the data in the entries of the dictionary given by `get_splits()`, so it's just the left side of the splits. To get the right side we take the whole data of the node and get rid of all the rows that are in the left split (`subset`).  
So now we can get the best split by looping over all possible splits, ad returning the one with the highest value of $$\Delta i$$:

```python
def get_best_split(self):
    all_splits = self.get_splits()
    delta_is = {}

    for key, split in all_splits.items():
        delta_is[key] = self.get_delta_i(split)

    return max(delta_is, key=delta_is.get)
```

# Building the tree

Ok so now for the fun part, we are going to build the tree recursively. To do that we choose the best split at our root node, and then apply the split function on each of these subtrees, and in turn each of the splits in these split will also be split... And this needs to keep happening until some conditions are met: the stop condition.  
So when do we want to stop splitting the data at a given node? Well the simplest answer would be to stop when the node is pure (or when there are no more possible splits), so it contains data points of only one class. However since, as we said in previous parts, we want to avoid overfitting we will also add some other stop conditions:

- Stop splitting when the leaf has under a certain amount of data points
- Stop splitting when the tree gets a certain depth.

To implement these two stopping options we need to add more parameters to our `Node` class, so let's modify ou `__init__` method:

```python
def __init__(self, data, target, min_samples_leaf=10, max_depth=3, level=0):
    self.data = data # Dataframe with features and target
    self.target = target # name of target
    self.min_samples_leaf = min_samples_leaf
    self.max_depth = max_depth
    self.level = level
    self.left = None
    self.right = None
    self.prediction = None
```

Ok so here I have just added parameters to determine when we want our tree to stop splitting, as well as a `level` value that is just going to allow us to keep track of the depth of a given node in the tree. Ok so now we have these parameters we need to implement methods that allow us to check if any of the stopping conditions are met:

```python
    def is_pure(self):
        return len(self.data[self.outcome].unique()) == 1

    def too_small(self):
        len(self.data) <= self.min_samples_leaf

    def too_deep(self):
        return self.level >= self.max_depth

    def no_splits(self):
        return self.get_splits() == {}
```

So if any of these methods return true we will stop splitting. Ok so now we have defined our stop condition we can write up our recursive splitting method:

```python
def split(self):

    if self.is_pure() or self.too_deep() or self.no_splits() or self.too_small():  # stop condition
        # we set the prediction value of this terminal node
        self.prediction = self.data[self.outcome].value_counts().idxmax()
        return

    # find best split
    best_split = self.get_best_split()

    # get split info
    self.split_feature = best_split[0]
    self.split_value = best_split[1]
    self.split_type = best_split[2]

    # get the actual right and left subsets to pass on to child nodes
    if self.split_type == 'categorical':
        left_data = self.data[
            self.data[self.split_feature] == self.split_value]
        right_data = self.data[
            self.data[self.split_feature] != self.split_value]

    elif self.split_type == 'numerical':
        left_data = self.data[
            self.data[self.split_feature] <= self.split_value
        ]
        right_data = self.data[
            self.data[self.split_feature] > self.split_value
        ]
    else:
        raise ValueError(
            'splits can be either numerical or categorical'
            )

    # get parameters to pass on to child nodes
    child_params = {
        'outcome': self.outcome,
        'parent_node': self,
        'min_samples_leaf': self.min_samples_leaf,
        'max_depth': self.max_depth,
        'level': self.level +1
    }

    # create child nodes and point to them from this node
    self.left = Tree(left_data, **child_params)
    self.right = Tree(right_data, **child_params)

    # split child nodes
    self.left.split()
    self.right.split()

    return
```

Ok so it might seem like a long function but it is actually quite simple, We just keep splitting the data with the best possible split (maximizing $$\Delta i$$), and if one of our stop conditions is met we get the prediction that this node will make: the most frequent class in the node.

All right we're done with the important bits, let's test our program out, and see what kind of trees we get, to be able to see what tree we have I blatantly ripped off <a></a>[this StackOverflow answer](https://stackoverflow.com/a/54074933/8650928) which gives us super nice trees! And I added a `value` property for my nodes where I put a string describing the split if the node is a split node, and the predicted class if the node is a leaf node.  
and if we try out our code with the iris data we get:

```python
>>>from sklearn.datasets import load_iris

>>>iris = load_iris()
>>>iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
>>>iris_df['species'] = iris['target']
>>>iris_df['species'] = iris_df['species'].apply(lambda i: iris['target_names'][i])

>>>tree_iris = Tree(iris_df, 'species', max_depth=3)
>>>tree_iris.split()
>>>tree_iris.display()

     ____petal length (cm) <= 1.9________________________
    /                                                    \
 (setosa)                               ______petal width (cm) <= 1.7______
                                       /                                   \
                                  (versicolor)                        (virginica)
```

Hey that tree looks super familiar, yay it's the exact same one than the in previous parts, our method worked! how about if we want a deeper tree?

```python
>>>tree_iris = Tree(iris_df, 'species', max_depth=4)
>>>tree_iris.split()
>>>tree_iris.display()

     ____petal length (cm) <= 1.9____________________________________________________________
    /                                                                                        \
 (setosa)                                                  _______________________petal width (cm) <= 1.7________________________
                                                          /                                                                      \
                                        ______petal length (cm) <= 4.9______                                    _____petal length (cm) <= 4.8______
                                       /                                    \                                  /                                   \
                                  (versicolor)                         (virginica)                        (virginica)                         (virginica)
```

We get a tree that's one level deeper. So everything seems to be working fine. However in our iris dataset we only have numerical data, lengths and widths, so we don't really know if our tree building method works with categorical data. So to do this I'm going to use the golfing dataset which has a certain number of features, and the target value is if a game of golf is played or not. This dataset is very small so I can show you all of it here:

| id  | outlook  | temperature | humidity | windy | play |
| --- | -------- | ----------- | -------- | ----- | ---- |
| 1   | sunny    | 85          | 85       | FALSE | no   |
| 2   | sunny    | 80          | 90       | TRUE  | no   |
| 3   | overcast | 83          | 86       | FALSE | yes  |
| 4   | rainy    | 70          | 96       | FALSE | yes  |
| 5   | rainy    | 68          | 80       | FALSE | yes  |
| 6   | rainy    | 65          | 70       | TRUE  | no   |
| 7   | overcast | 64          | 65       | TRUE  | yes  |
| 8   | sunny    | 72          | 95       | FALSE | no   |
| 9   | sunny    | 69          | 70       | FALSE | yes  |
| 10  | rainy    | 75          | 80       | FALSE | yes  |
| 11  | sunny    | 75          | 70       | TRUE  | yes  |
| 12  | overcast | 72          | 90       | TRUE  | yes  |
| 13  | overcast | 81          | 75       | FALSE | yes  |
| 14  | rainy    | 71          | 91       | TRUE  | no   |

That's it, that's the whole dataset, but you see here we have a nice mix of categorical and numerical data. Ok so let's see how our CART implementation handles this:

```python
>>>import pandas as pd
>>>data_mixed = pd.read_csv('data_mixed.csv', header=0, index_col=0)

>>>tree = Tree(data_mixed, 'play', max_depth=4)
>>>tree.split()
>>>tree.display()

    __outlook = overcast___________________________________
   /                                                       \
 (yes)                                ______________humidity <= 80______________
                                     /                                          \
                           __temperature <= 65___                     __temperature <= 70__
                          /                      \                   /                     \
                         (no)                  (yes)               (yes)                  (no)
```

Yay everything works!

You might have noticed that we only have classification trees in this example, and you'd be right. I haven't implemented the regression part yet because I'm too lazy but it would be exactly the same, but you would need to add an RSS function that you could plug in the `get_delta_i()` method and in the `split()` method, when a leaf node is reached set the prediction value to the mean of the dataset outcomes instead of the most frequent one. So I'll put it in eventually but I won't make a separate post on that. All of the code is on my <a></a>[github](https://github.com/zlanderous/CART-python) so you can play with it if you want.  
One last thing, we haven't implemented the full CART algorithm because there is no pruning method to avoid overfitting, but this will come in a future part, so stay tuned!.
