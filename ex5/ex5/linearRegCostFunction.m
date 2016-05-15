function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Copied from ex1

H = X * theta;
J = sum((H - y) .^ 2) / (2 * m);

% Copied from ex3
% See https://www.coursera.org/learn/machine-learning/module/mgpv7/discussions/0DKoqvTgEeS16yIACyoj1Q
tmp = theta;
theta(1) = 0;
J += lambda * theta' * theta / (2 * m);


% Vectorized, after reading this advice:
% https://www.coursera.org/learn/machine-learning/discussions/GVdQ9vTdEeSUBCIAC9QURQ

% Remember that theta(1) = 0;
grad = X' * (H - y) / m + theta * lambda / m;

theta = tmp;









% =========================================================================

grad = grad(:);

end
