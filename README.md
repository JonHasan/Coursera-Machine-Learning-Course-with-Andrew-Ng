# Coursera-Machine-Learning-Course-with-Andrew-Ng


Introductory Machine Learning Course from Stanford  - My first MOOC started 6 years ago. Now completed after all this time... One more victory.....

Goes over the basic Machine Learning Techniques that are common in the field and applied them to a wide variety of different Cases.

### Note of thanks: 

Of course, Professor Ng deserves thanks for making this course and sharing his knowledge with all of his students world wide. Thank you. 

I would like to thank the mentor Tom Mosher for his very helpful tutorials which have been of great assistance in finishing these project assignments. I will be placing pictures of his helpful hints in the future, I hope. 


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

### Cost function for Linear Regression 

How do we know if the hypothesis is getting closer to the correct response values according to the dataset. Well in this case its just determinng the Sum of the square or errors. It penalizes very heavily results that are far away from the true value. 

$J = {1}{2*m} * \Sigma((h_{\theta}*X - y)^2)$

With poor theta parameters one would get a very bad fit of data as shown below. Graph was made by Professor Ng. 

![image](https://user-images.githubusercontent.com/20827630/184058843-31af0f2c-92db-4bef-9458-78f0b8c216cf.png)

Source: Source: Machine Learning Course by Andrew Ng - Week 1 - Cost Function- Intuition || 

The theta parameters are modified so that the distance between the points is minimized. This results in a graph shown below that was made by Professor Ng. 

![Professor Ng's iteration of Linear Regression](https://user-images.githubusercontent.com/20827630/184058679-73a12e71-223c-4727-ba8d-f3bb3881a740.png)

Source: Machine Learning Course by Andrew Ng - Week 1 - Cost Function- Intuition || 

### Gradient Descent for Linear Regression

In order to fit the hypothesis to the response variable, it is necessary to change the theta parameters. This is done with a technique called gradient descent. It uses the following formula: 

$\theta_{n} = \theta_{n} - \alpha * \Sigma(h_{\theta} * X - y)$

This can be easily extrapolated to as many theta parameters as necessary. What happens is that the theta parameters in the hypothesis function are updated until the function matches the response variable as closely as possible. 

### Results of Linear Regression project: 

![image](https://user-images.githubusercontent.com/20827630/184282376-a2b10c8f-26d1-41f1-8474-56685901377d.png)



## Week 3: Logistic Regression 

## Explanation: 

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

$J = \frac{1}{m} * \Sigma(Cost(h_{\theta})$

Where

$Cost(h_{\theta}) = log(h_{\theta}*X)$ if y = 1

$Cost(h_{\theta}) = -log(1-h_{\theta}*X)$ if y = 0

The reason why this Cost function is used is it was just a creative exercise in order to achieve the desired outcome. What function would allow me to penalize incorrect classifications extremely and when the classifications get closer the cost falls extremely low the closer it gets to the correct value. The functions above seem to do the job correctly so these are used. 

Professor Ng elegantly shows the Cost function as a consequence of the hypothesis function for both scenarios. 

![Professor Ng's graphs for logistic Cost function](https://user-images.githubusercontent.com/20827630/184055145-3c3ccf73-702e-4248-af51-e33b6df125de.png)

Source: Machine Learning Course By Andrew NG - Week 3 Reading: Cost Function

So combining both of these functions will allow for computing the appropriate cost for the hypothesis function. 

### Gradient Descent for Logistic Regression

### Multiclass Classification 

TBD

### Results of Logistic Regression Project 

![image](https://user-images.githubusercontent.com/20827630/184286261-b0e991ef-ffab-4fad-8c00-95581b298505.png)



## Week 4: Multi-Class Classification and Neural Networks 

### Explanation:

This week went over one of the most recent additions to the machine learning engineeers toolkit called a neural network. It is an algorithm inspired by the inner workings of the brain. It is believed that the brain is using something called "The Master Algorithm" which allows different parts of the brain that are responsible for one function i.e. like hearing and training that brain tissue to be used for seeing things. The brain is comprised of a particular type of cell called a neuron. It has dendrite that accept inputs and sends the result through to the axon which is the output. The algorithm shown in this week attempts to recreate this powerful functionality. 

The nerual network is a universal function approximator that can create non linear decision boundaries and is inspired by the way the brain learns. 

Between the input and output, we can have hidden layers that can allow for the matching of more complicated non-linear decision boundaries. 

Here is a simplified version of the progression of the neural network. 

$[x1x2x3] -> [] -> h_{\theta}*x$

The inputs x1x2x3 are like the dendrites in the neurons, the inputs are then processed within other hidden networks of neurons and then output in the hypothesis or the axon of the last neurons. 

So what is in the empty brackets? It is the activation functions that show the intermediate outputs. These are [a1a2a3]. What are these activation units though? 

They are the summations of all the results from the previous neurons which are then fed into a sigmoid function. 

Neural networks can be represented with the following type of diagram. 

![image](https://user-images.githubusercontent.com/20827630/184502416-cc763568-df40-4f82-81ff-9c911e4df213.png)

Source: Machine Learning with Andrew Ng - Week 4 - Neural Networks - Lecture_slides 

In order to perform the prediction phase it is necessary to perform something called a feedForward operation. 

All this operation is doing is multiplying the inputs with the Theta Matrices, activating the results with sigmoid function and adding bias units whenever necessary. 

The resulting hypothesis function will appropriately classify all the examples with appropriate label. 

The equations used in the feedforward operation are shown below for the example network. 

This is shown below and was taken from the course. 

![image](https://user-images.githubusercontent.com/20827630/184272996-f8a6c9b9-48b6-441e-9b2e-b4fc5063191c.png)
 
Source: Machine Learning with Andrew Ng - Week 4 - Neural Networks - Model Representation l

### Cost function for neural networks in classification 

The Cost function used by neural networks for classification is the same as the one used for Logistic Regression. It is shwon below 

$J(\Theta) = \frac{-1}{m} * \Sigma(y^{i} * log(h_{\theta} * X) + (1 - y^{i}) * (1 - log(h_{\theta} * X)))$

This is applied to each of the output nodes to see which class accurately represents the training example in the X matrix. 

### Gradient for neural networks in classification 

The gradient is the derivative of the Cost function with regard to the theta parameters. It is shown below. 

$\frac{d(J_{\Theta})}{d_{\Theta_{j}} = -\frac{1}{m} * \Sigma(h_{\theta} * X - y) * X$

This will allow you to choose the optimal theta parameters that will minimize the Cost Functions for each class label. 

### Results of Neural Network MultiClassification Project 

![image](https://user-images.githubusercontent.com/20827630/184469799-1eca7fac-b6a7-4e63-a3fc-e0ae305497c7.png)



## Week 5: Neural Network Learning 

## Explanation: 

The previous week taught me how to perform a classification task with pre-made Theta matrices and then sending the inputs through these theta matrices and activating the layers in order to find the correct labels for each feature. However, what if the matrices are not the ones that give the highest accuracy from the start? 

This requires the use of a technique called backpropagation which allows for the modifying of the theta parameters. 

## Cost Function for Neural Network Learning: 

$$ \qquad \sum_{i=1}^m \sum_{k=1}^k [y_{k}^{i} * log(h_{\theta}*(x^{i})_{k} + (1-y_{k}^{i}) * (1 - log(h_{\theta}*(x^{i})_{k})]  + \sum_{l=1}^{l + 1} \sum_{i=1}^sl \sum_{j=1}^{sl + 1} (\Theta_{j}^{l})^2)  $$

It looks like the Cost function is adding up all of the examples for all of the labels. 

### Gradient for Neural Network Learning 

The gradient is calculated differently this time. Instead of using the gradient used in Week 4, it is now going to be done using a tchnique called BackPropagation. 

#### Explanation of BackPropagation: 

Here is a schematic showing a brief view of BackPropagation in action. The diagram was taken from the Machine Learning Course. 

![image](https://user-images.githubusercontent.com/20827630/184551188-022ba661-3344-4a50-ad57-3509ece34c93.png)

Source: Machine Learning with Andrew Ng - Week 5 - Neural Networks Learning - BackPropagation Intuition

Lets say you calculate the activation results for each layer in a neural network. The activation layer for the last layer is the hypothesis you are comparing to the true value y. Determine how close the hypothesis and the true value is by subtracting the results from each other: 

$\delta_{L} = a(L) -  y$

Now it is required to send this difference back through the layers. The equations used for that are 

$\delta(3) = (\Theta^{3})^{T} * \delta^{4} .* g'(z^{3}) $

$\delta(2) = (\Theta^{2})^{T} * \delta^{3} .* g'(z^{2}) $

g' is the sigmoid gradient which is the derivative of the sigmoid function. My guess it will be a horizontal line at the beginning and will be a straight line when it curves up and down. 

It is then possible to add up all of these errors and get the partial derivative. 

$\Delta_{i,j}^{l} = \Delta_{i,j}^{l} + a_{j}^{l} * \delta_{i}^{l+1}$

So I can use this derivative to guide the Theta Matrices in the right direction. 

### Results of Neural Network Learning Project 

![image](https://user-images.githubusercontent.com/20827630/184470456-560fb781-27d8-4d60-88be-59b68139a62f.png)


## Week 6: Regularized Linear Regression for Bias/Variance 

Explanation: 

So you have made an algorithm, (in this case a linear regression with a regularization parameter), it now time to optimize the algorithm to see which parameters give the best model. 

What if the model you are making doesn't accurately represent the test data? Well you can try a battery of different modifications that are suggested: 

- Getting more training examples
- Trying a smaller amount of features
- Getting additional features
- Trying polynomial features
- Increasing or decreasing $\lambda$

So when the hypothesis does as well with the prediction as you would like, there is still an issue here. The hypothesis is tested with this data. So if you get perfect predictions, it might mean that the model will only work with that dataset. Its like teaching a student with questions in A hw set and then placing those exact same questions in the final exam. You want to test the model's ability to generalize. 

With this in mind the next phase of testing may begin. This involves creating different datasets for testing. This can be made from the original dataset by performing a 70:30 split. 70% of the data will be used to train the hypothesis and the other 30% is used to evaluate the hypothesis. 

There is a new metric that needs to be considered called the Test Set Error and was summarized by Professor Ng himself. 

![image](https://user-images.githubusercontent.com/20827630/184552158-571517d0-9514-4ded-b5d5-6be1a519df20.png)


Source: Machine Learning with Andrew Ng - Week 6 - Evaluating a Hypothesis


However, there is another set the Cross Validation set that will allow for choosing the appropriate amount of polynomial features. First Theta is chosen by training set, the different polynomial degrees are tested on CV set and the lowest error for that one is used on the test set. We can now determine whether there is High Bias(Underfitting) or High Variance(Overfitting). The different scenarios are shown in a figure from the course. 

![image](https://user-images.githubusercontent.com/20827630/184552869-dec38150-0f98-4a98-af62-ac1c4cde63ea.png)

Source: Machine Learning with Andrew Ng - Week 6 - Diagnosing Bias vs Variance 


So how can the regularization term help with this? Well as you have seen from previous lessons, regularization can destroy the effects of polynomial features. So what $\lambda$ value will get the correct model? Well as the professor said, we can try a whole different array of regularization parameters and see which model is correct. 

Professor Ng summarizes this very well with the following diagram from his course. 

![image](https://user-images.githubusercontent.com/20827630/184553380-5379c104-6aa8-4743-8f02-1358a3b863db.png)

Source: Machine Learning with Andrew Ng - Week 6 - Regularization Bias and Variance 

You can choose the right $\lambda$ and use it to make the best hypothesis. 


This is all the information that will be exercised in the project for this Week. 

### Results

![image](https://user-images.githubusercontent.com/20827630/184470566-a8f478f1-0ebe-4ec7-888c-6d3ef039afed.png)


## Week 7: Support Vector Machines 

## Explanation: 

Support Vector Machines are a black-box algorithm that uses an optimization objective that is a little more difficult then the previous weeks. 

The optimization objective is a play on the optimization objective for logistic regression. 

To understand this modified objective, it is required to understand a modified view of the sigmoid function. 

Instead of z in the sigmoid function, replace it with its other representation which is $\Theta^{T}*X$

It is seen that for both cost functions that comprise logistic regression, when y = 1, the cost function gives a proper classification when $\Theta^{T} * X $ is very large. For y = 0, the other cost function shows a proper classification when $\Theta^{T} * X $ is very small. So how can can we use this information? Well, we can simplify the logistic equation into a piecewise function. 

The Cost function for y = 1 can be reduced to a line with a negative slope connected to a line with 0 slope. 

The Cost function for y = 0 can be reduced to a line with a positive slope that extends from a line with 0 slope. 

This is shown in the diagram taken from the course. 

![image](https://user-images.githubusercontent.com/20827630/185022952-5d3a71c9-bfbc-4943-b448-4fc6ca2716f8.png)

Source: Machine Learning with Andrew Ng - Week 7 - Support Vector Machines - Lecture Slides 

So now instead of the logistic Cost function which uses exponentials, we can now use these modified cost functions which don't have exponentials. 

## Cost Function for Support Vector Machines 

min C $ \Sigma_{i=1}^{m}[y^{(i)} * cost_{1}($\Theta^{T}*x^{i}) + (1 - y^{(i)}) * cost_{0}($\Theta^{T}*x^{i})] + \frac{1}{2} * \Sigma_{i=1}^{n} \theta_{j}^{2} $


### Results of Support Vector Machine Project 

![image](https://user-images.githubusercontent.com/20827630/184470808-cf0fb79e-3c8f-44fb-a0f7-3115078c8c02.png)


## Week 8: K-means Clustering and PCA

### Explanation: 

### Cost function for K-means and PCA

### Results of K-means Clustering and PCA Project 

![image](https://user-images.githubusercontent.com/20827630/184471056-49379698-d58d-4381-933a-d09ddbf96a75.png)

## Week 9: Anamoly Detection and Recommender Systems 

### Explanation: 

### Cost function for Anomaly Detection and Recommender Systems. 

### Results of Anomaly Detection and Recommender systems 

![image](https://user-images.githubusercontent.com/20827630/184471099-8c76f324-9498-4f65-be91-87334682f227.png)


