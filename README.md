# Coursera-Machine-Learning-Course-with-Andrew-Ng
Introductory Machine Learning Course from Standford 

Goes over the basic Machine Learning Techniques that are common in the field. 

## Week 2: Linear Regression 

Purpose: Learn about a Machine Learning Technique that will draw a trendline that can relate a set of features to an output. 
### Explanation
In the course, Linear Regression was one of the first techniques introduced in the course.

The technique attempts to recreate a function through iteration that can match the response variable that is continuous. 

If we had a dataset on housing prices for example, one would find that the price of the house varies depending on certain key factors. Some of these factors are things like square footage, number of bathrooms, number of bedrooms, Number of floors etc. 

| Size    | Number of bedrooms      | Number of floors| Price($1000)   
| ------------- | ------------- | --------    | ---------
| 2104      | 1    | 45   | 460
| 1416         | 2   | 40   | 232

Source: Machine Learning on Coursera - Video: Multiple Features from Week 2 

The question was then put forth. 

Is it possible to create a relationship between these stated "features" i.e. Size, num of bedrooms etc and the price?

It is indeed possible and that was the focus of the project. The way this is done is by forging a polynomial equation that can generate the price response. This polynomial equation is referred to in the series as a hypothesis. 

$$h_{0} = \theta_{0} + \theta_{1}*x_{1} + \theta_{2}*x_{2} + \theta_{3}*x_{3} +\theta_{n}*x_{n}$$

This hypothesis attempts to accurately predict the response variable given a feature space.

### Modifying theta with gradient descent 

In order to fit the hypothesis to the response variable, it is necessary to change the theta parameters. This is done with a technique called gradient descent. It uses the following formula: 

$\theta_{n} = \theta_{n} - \alpha * \Sigma(h_{\theta} * X - y)$

This can be easily extrapolated to as many theta parameters as necessary. What happens is that the theta parameters in the hypothesis function are updated until the function matches the response variable as closely as possible. 

## Week 3: Logistic Regression 

Logistic Regression was the next important technique taught in the course. 

Instead of trying to predict a continuous variable, this technique tries to predict a categorical variable i.e. Cancer or No Cancer, Lend Money vs Don't Lend Money, Happy vs Sad etc. Its either one or the other. 

The professor discussed how a naive attempt to solve the problem was to use linear regression and just limit the bounds of the response variable to between 0 and 1. However, the response is categorical while linear regression will potentially give results that are between 0 and 1. If you are building a cancer classifier, I imagine you don't want to get a response of .5. That is a meaningless result. 

So what function can be used if not the polynomial expression used in Week 2? 

Looks like the function referred to as the sigmoid function will be of use. 

This function is shown below. 

$g(z) = \frac{1}{1+e^{-z}}$

![Logistic Function](https://user-images.githubusercontent.com/20827630/184052065-e4529d1c-794e-426a-9997-6d3926f4caa8.png)

Source: Machine Learning Course By Andrew NG - Week 3 Reading: Hypothesis Section

This function will map any real value number into a response from 0 to 1. Notice that z has a very small range where it gives a non-integer input. This way, it will be able to classify most of the examples into the categorical bins of 0 and 1.

The hypothesis function therefore becomes a probability that an input will result in a categorical variable response. i.e. P(Y = 1) = .70

However, we still need a way of forcing the function to take a stand. We need to enforce a decision boundary. If it is on one side of the boundary, it will go into one bin. If it is on the other side of the boundary, it will go to the other bin. 

The decision boundary for the logistic function was decided to be .5. If the hypothesis is greater than .5, output the categorical value of 1. Else output categorical variable of 0. The input in octave is just the Theta Matrix transpose multiplied by the X matrix. $\Theta * X$. Note that the input does not have to be linear. It can be whatever shape fits the data. 

### Cost Function for Logistic Regression 

So how do we determine if we are accurately predicting the response variables? Well the following cost function is used. 

$Cost(h_{\theta} = log(h_{\theta}*X)$

\begin{equation}
 f(x)=\begin{cases}
    1, & \text{if $x<0$}.\\
    0, & \text{otherwise}.
  \end{cases}
 \end{equation}

## Week 4: Multi-Class Classification and Neural Networks 

## Week 5: Neural Network Learning 

## Week 6: Regularized Linear Regression for Bias/Variance 

## Week 7: Support Vector Machines 

## Week 8: K-means Clustering and PCA

## Week 9: Anamoly Detection and Recommender Systems 
