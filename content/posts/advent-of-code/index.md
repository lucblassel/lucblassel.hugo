---
title: "Advent of Code 2021 - Problem 7"
date: 2021-12-07T18:44:59+01:00
latex: true
summary: |
    As I was doing the 7th problem of the 2021 Advent of Code, I realized the mean and median were popping up in the optimal solutions. There is a fairly simple explanation to this... 
---

I am currently doing the [advent of code 2021](https://adventofcode.com/2021) programming challenge, which (at the time I'm writing this), is on day 7.  

I got the motivation to write this post from [this tweet](https://twitter.com/QuentinLclrc/status/1468195574776274951?s=20) that had some questions about the solution.  

I am going to attempt to explain my approach to this problem and why we find this particular solution, so if you have not seen the problem I encourage it to try it yourself before reading this *(have fun!)*. 

# The first problem
## What's the problem setting ? 
You are in a submarine, deep under the ocean, when a giant whale decides to eat you. suddenly a bunch of crabs in tiny crab-submarines appear and decide to help you. To be able to help you the crabs need to align themselves, however, being crabs after all, they can only move sideways. The crab submarines have a limited amount of fuel, so you need to find the where the crabs should align in order to expend the smallest amount of fuel as possible. The puzzle input is the horizontal position of the crab.  
In the given example there are n crabs at positions `16,1,2,0,4,2,7,1,2,14`. We can represent this with a simple diagram, making a grid with 16 columns for all the possible horizontal crab position, a row per crab and marking the position of the crab with `C`:  
```
  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
  -------------------------------------------------
  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  C
  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .  .
  C  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  .  .  .  .  C  .  .  .  .  .  .  .  .  .
  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  .  .  .  .  .  .  .  .  .  .  .  C  .  .
```

Let's say we want them to align at column 3, then our final crab positions will be:  
```
  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
  -------------------------------------------------
  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .
  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .  .  .
```
Our goal is to find the column in which to align them, for which the least amount of fuel will be consumed by the crabs. If we choose column \\(3\\), then the first crab will have to spend \\(16 - 3 = 13\\) units of fuel, the second crab will need \\(3 - 1 = 2\\) units of fuel and so on.  

## A brute force approach
This is very simple to solve with a brute force approach, we define a function to compute the cost of moving all the crabs to a column, apply this function to all possible columns and see which one is smallest: 
```python
def compute_fuel_usage(crabs, destination):
    fuel_cost = 0
    for crab in crabs:
        fuel_cost += abs(crab - destination)
    return fuel_cost

crabs = [16, 1, 2, 0, 4, 2, 7, 1, 2, 14]

min_fuel = -1
min_pos = 0
for destination in range(min(crabs), max(crabs) + 1):
    usage = compute_fuel_usage(crabs, destination)
    if usage < min_fuel or min_fuel < 0:
        min_fuel = usage
        min_pos = destination

print(f"Crabs should move to position {min_pos}, using {min_fuel} fuel")
```

This will give us the *(correct)* answer of \\(2\\) with a fuel cost of \\(37\\). Now this is a fine solution for the test input but it will be very inefficient if we have more crabs and if they are more spread out. 

## An analytical solution
The first thing I thought about is exploring significant points like the [mean](https://en.wikipedia.org/wiki/Mean) or the [median](https://en.wikipedia.org/wiki/Median) of our crab positions. In our example above the mean is not the optimal solution however the median is equal to \\(2\\), which ***is*** optimal. After a bit more saerching I found the [geometric median](https://en.wikipedia.org/wiki/Geometric_median), which is *" the point minimizing the sum of distances to the sample points"* of a dataset. This is exactly what we are looking for, and per the wikipedia article, in a 1-dimensional setting *(like ours)* it is equal to the median. **Bingo** we have a very simple way of getting the optimal position: just take the median of our dataset, it is super quick and we only need to compute 1 thing. Our new code becomes: 

```python
from statistics import median

min_pos = median(crabs)
min_fuel = compute_fuel_usage(min_pos)

print(f"Crabs should move to position {min_pos}, using {min_fuel} fuel")
```

Through a bit of exploration I think anybody would have realized that the median is actually the optimum, however we can also derive it analytically and prove why the optimum is actually the median.  
Our problem is in essence an optimization problem, we have a cost *(the fuel usage)* which depends on a parameter *(the column in which to line up)* and we want to minimize that cost w.r.t the chosen parameter. This is a very common framework and if the cost is a simple function we can derive an analytical solution to our problem.  
First let's define some terms: 
 - I will call \\(n\\) the number of crabs
 - \\(x_i\\) is the horizonal position of the \\(i^{th}\\) crab. So in our example above \\(x_0=16\\) and \\(x_1=1\\).
 - \\(\theta\\) is the x-position at which we want the crabs to align
 - \\(\theta^\*\\) is the optimal value of \\(\theta\\), so in our example above \\(\theta^*=2\\)
 - \\(C(\theta)\\) is the cost function *(i.e the amount of fuel used)* for a given value of \\(\theta\\). In our example above \\(C(2) = 37\\).

The first step in any optimization function is to define the cost function. In our case it's quite simple, our cost is the sum of all the absolute differences between each crab and the chosen value of \\(\theta\\):
$$
C(\theta) = \sum_{i=1}^n|x_i - \theta|
$$

Now that we have this cost function how are we supposed to find the minimum you might ask ? If you have taken a calculus class this will no doubt sound familiar, we need to compute the derivative of \\(C(\theta)\\) and find the value of \\(\theta\\) for which it is equal to 0. That simple!  
Now this cost function is simply a sum so it will be quite easy to compute its derivative. First we need to break it up, we can define the absolute value \\(|x_i - \theta|\\) as follows:

{{<longmath>}}
$$
|x_i - \theta| = 
\begin{cases}
\theta - x_i &\text{if }x_i \leq \theta \\
x_i - \theta &\text{if }x_i > \theta\\
\end{cases}
$$
{{</longmath>}}

So our cost function becomes: 

{{<longmath>}}
$$
\begin{align}
C(\theta) &= \sum_{x_i \leq \theta} \big(\theta - x_i\big) + \sum_{x_i > \theta} \big(x_i - \theta\big) \\
    &= \sum_{x_i \leq \theta} \theta - \sum_{x_i \leq \theta} x_i + \sum_{x_i > \theta} x_i - \sum_{x_i > \theta} \theta\\
\end{align}
$$
{{</longmath>}}

If we take the derivative of this expression w.r.t \\(\theta\\) both sums which only contain \\(x_i\\) will be reduced to 0, and since the derivative of the sum is equal to the sum of the derivatives it is eay to compute the rest of the terms:  

{{<longmath>}}
$$
\begin{align}

\frac{\partial\,C(\theta)}{\partial\theta} &= \frac{\partial}{\partial\theta}\bigg(\sum_{x_i \leq \theta} \theta - \sum_{x_i > \theta} \theta\bigg)\\
    &= \sum_{x_i \leq \theta} \frac{\partial\,\theta}{\partial\theta} - \sum_{x_i > \theta} \frac{\partial\,\theta}{\partial\theta}\\
    &= \sum_{x_i \leq \theta} 1 - \sum_{x_i > \theta} 1\\
\end{align}
$$
{{</longmath>}}

There we have a simple definition of the derivative of our cost function w.r.t to the parameter we wish to optimize. Now to find \\(\theta^*\\) we simply need to find where the derivative is equal to 0:  
{{<longmath>}}
$$
\begin{align}
& \frac{\partial\,C(\theta^*)}{\partial\theta} = 0 \\
\Leftrightarrow & \sum_{x_i \leq \theta} 1 - \sum_{x_i > \theta} 1 = 0 \\
\Leftrightarrow & \sum_{x_i \leq \theta} 1 = \sum_{x_i > \theta} 1
\end{align}
$$
{{</longmath>}}

You will notice that for our derivative to be equal to 0 we need to have as many points \\(\leq\theta^* \\) as there are points \\(>\theta^* \\), which is exaclty the definition of the median.  
Take a minute to work through it by yourself and make sure I haven't made any mistakes but by now it should make sense *why* the median is the optimal solution in our problem.

# The second problem
As is usual in advent of code problems, getting the correct answer in the first problem unlocks a second, harder, problem which builds upon the first. In our case the setting is the same except that we misunderstood the fuel usage of the crab submarines. Instead of using up 1 unit of fuel per unit of distance, they use up an additional unit of fuel for each unit of distance covered. For instance, a crab submarine that will travel 3 units of distance will use up \\(1 + 2 + 3 = 6\\) units of fuel. 

The brute force approach still works fine on the smaller test data, we simply need to update our `compute_fuel_usage` function:
```python
def compute_new_fuel_usage(crabs, destination):
    fuel_cost = 0
    for crab in crabs:
        for i in range(1, abs(crab - destination) + 1):
            fuel_cost += i
    return fuel_cost
```

But once more this approach is super inneficient as we grow the number of crabs and the number of columns. So here it is better to consider this as an optimization problem once more. We simply need to redefine the cost function to fit our problem and we can follow the exact same steps as before to find \\(\theta^*\\). And a way to write this cost is as follows:  

$$
C(\theta) = \sum_{i=1}^n\bigg( \sum_{j=1}^{|x_i - \theta|} j \bigg)
$$

Where both sums corresponds to both loops of our `compute_new_fuel_usage` function. Now this cost function does not see, very derivative friendly so lets try to break it up. The first thing we need to know is that there is a forumla for the sum of the \\(k\\) first natural integers:  

$$
\sum_{i=1}^{k} i = \frac{k \cdot (k + 1)}{2}
$$

So in our case we can rewrite the cost function as:  
{{<longmath>}}
$$
\begin{align}
C(\theta) &= \sum_{i=1}^n\bigg( \frac{|x_i - \theta| \cdot (|x_i - \theta| + 1)}{2} \bigg) \\
    &= \sum_{i=1}^n\bigg( \frac{|x_i - \theta||x_i - \theta| + |x_i - \theta|}{2} \bigg) \\
    &= \frac{1}{2}\sum_{i=1}^n\big((x_i - \theta)^2 + |x_i - \theta|\big) \\
    &= \frac{1}{2}\bigg( \sum_{i=1}^n(x_i - \theta)^2 + \sum_{i=1}^n|x_i - \theta| \bigg) \\
    &= \frac{1}{2}\bigg( \sum_{i=1}^n \big(x_i^2 - 2\cdot x_i\cdot\theta + \theta^2\big) + \sum_{i=1}^n|x_i - \theta| \bigg) \\
    &= \frac{1}{2}\bigg(\sum_{i=1}^n x_i^2 - 2\sum_{i=1}^n x_i\cdot\theta + \sum_{i=1}^n \theta^2 + \sum_{x_i \leq \theta} \theta - \sum_{x_i \leq \theta} x_i + \sum_{x_i > \theta} x_i - \sum_{x_i > \theta} \theta\bigg)
\end{align}
$$
{{</longmath>}}

whoo that's big and scary, but when we take the derivative w.r.t to \\(\theta\\) all the sums that don't depend on \\(\theta\\) will disappear:  

{{<longmath>}}
$$
\begin{align}
\frac{\partial\,C(\theta)}{\partial\theta} &= \frac{1}{2}\frac{\partial}{\partial\theta}\bigg( - 2\sum_{i=1}^n x_i\cdot\theta + \sum_{i=1}^n \theta^2 + \sum_{x_i \leq \theta} \theta - \sum_{x_i > \theta} \theta\bigg) \\
    &= \frac{1}{2}\bigg( - 2\sum_{i=1}^n x_i + \sum_{i=1}^n 2\theta + \sum_{x_i \leq \theta}1 - \sum_{x_i > \theta}1 \bigg) \\
    &= - \sum_{i=1}^n x_i + \sum_{i=1}^n\theta + \frac{1}{2}\bigg(\sum_{x_i \leq \theta}1 - \sum_{x_i > \theta}1\bigg)
\end{align}
$$
{{</longmath>}}

Let's call the last term \\(R\\) for notational simplicity and we have our derivative: 

$$
\frac{\partial C(\theta)}{\partial\theta} = - \sum_{i=1}^n x_i + \sum_{i=1}^n\theta + R
$$

Now we just need to find the minimum:  

{{<longmath>}}
$$
\begin{align}
& \frac{\partial C(\theta^*)}{\partial\theta} = 0\\
\Leftrightarrow &\; - \sum_{i=1}^n x_i + \sum_{i=1}^n\theta^* + R = 0 \\
\Leftrightarrow &\;   \sum_{i=1}^n\theta^* = \sum_{i=1}^n x_i - R \\
\Leftrightarrow &\;   n\cdot\theta^* = \sum_{i=1}^n x_i - R \\
\Leftrightarrow &\;   n\cdot\theta^* = \sum_{i=1}^n x_i - R \\
\Leftrightarrow &\;   \theta^* = \frac{1}{n}\sum_{i=1}^n x_i - \frac{R}{n} \\
\end{align}
$$
{{</longmath>}}

Now we can assume that R divided by n is probably small if the dataset is well distributed [^1]. So our optimal value will be close to the mean of our crab positions: 

$$
\theta^* \approx \frac{1}{n}\sum_{i=1}^n x_i
$$

And it is indeed a good place to start, so we can rewrite our code to look at the points around the mean to find the optimum: 

```python
from statistics import mean

m = mean(crabs)
candidates = [m-1, m, m+1]
min_pos, min_fuel = 0, -1
for candidate in candidates:
        fuel = compute_new_fuel_usage(positions, candidate)
        if fuel < min_fuel or min_fuel < 0:
            min_fuel = fuel
            min_pos = candidate

print(f"Crabs should move to position {min_pos}, using {min_fuel} fuel")
```

This will allow us to find the optimal point much more efficiently!

# Parting words
I hope you understood the approach I took and if you are interested in my progress on advent of code 2021 you can check out my [github repo](https://github.com/lucblassel/AoC_2021/) for my python solutions (while I can solve the problems :p).  
Thanks for reading and if you notice any mistakes in my reasonning please don't hesitate to point them out to me!

[^1]: **N.B**: as pointed out [here](https://twitter.com/ArthurCarcano/status/1468360563713576964?s=20) the value of \\(R\\) is actually bounded:  
Since all \\(x_i\\) and \\(\theta\\) values are positive: \\( 0 \leq card(\{x_i\\leq\theta\}) \leq n \\) and \\( 0 \leq card(\{x_i>\theta\}) \leq n \\). Therefore:  
{{<longmath>}}
$$
\begin{gather}
                -n \leq \sum_{x_o\leq\theta}1 - \sum_{x_o>\theta}1 \leq n \\
\Leftrightarrow \frac{-n}{2} \leq R \leq \frac{n}{2} \\
\Leftrightarrow \frac{-1}{2} \leq \frac{R}{n} \leq \frac{1}{2} \\
\end{gather}
$$
{{</longmath>}}
This means that at most the optimum will be \\(1/2\\) off of the mean, therefore the integer optimum must be at the integer mean \\(\pm 1\\).
