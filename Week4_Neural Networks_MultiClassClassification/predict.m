function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%feedforward operation time. Perform activations between each layer and determine which row is which class

% Need to add a bias unit to transform into activation function layer 1

column_ones = ones(size(X,1),1);

a1 = [column_ones,X];

%Multiply by Theta 1 and get 'z2'

a2 = sigmoid(a1 * Theta1');

%apply activation function and add ones to turn into activation layer 2

a2 = [ones(size(a2,1),1), a2];

%Multiply the activation layer 2 by Theta 2, compute sigmoid and make a3

a3 = sigmoid(a2*Theta2');

%Acquire the vectors, first vector shows indices of max values and second is vector of predictions for each row in X.

[index p] = max(a3, [], 2);











% =========================================================================


end
