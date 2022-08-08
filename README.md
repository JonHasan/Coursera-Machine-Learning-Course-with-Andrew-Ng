# Coursera-Machine-Learning-Course-with-Andrew-Ng
Introductory Machine Learning Course from Standford 

Goes over the basic Machine Learning Techniques that are common in the field. 

## Week 2: Linear Regression 

Purpose: Learn about a Machine Learning Technique that will draw a trendline that can relate a set of features to an output. 
### Explanation
In the course, Linear Regression was one of the first techniques introduced in the course. If we had a dataset on housing prices for example, one would find that the price of the house varies depending on certain key factors. Some of these factors are things like square footage, number of bathrooms, number of bedrooms, Number of floors etc. 

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

In order to fit the hypothesis to the response variable, it is necessary to change the theta parameters. 

### Octave Implementation: 

So it is now desired to create the hypothesis in octave. I prefer to vectorize the solution whenever possible and as recommended by the staff. So if the Theta weight vector is represented as follows: $$[\theta_{0}, \theta_{1}]$$. 

## Week 3: Logistic Regression 

## Week 4: Multi-Class Classification and Neural Networks 

## Week 5: Neural Network Learning 

## Week 6: Regularized Linear Regression for Bias/Variance 

## Week 7: Support Vector Machines 

## Week 8: K-means Clustering and PCA

## Week 9: Anamoly Detection and Recommender Systems 
