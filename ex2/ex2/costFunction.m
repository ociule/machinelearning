function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

H = sigmoid(X * theta);
J = sum(-y .* log(H) - (1 - y) .* log(1 - H)) / m;

% First try: unvectorized
%for i = 1:size(theta)
%    grad(i) = 1/m * sum((sigmoid(X * theta) - y) .* X(:, i));
%end

% Second try, with arrayfun
%f = @(i) 1/m * sum((sigmoid(X * theta) - y) .* X(:, i));
%yy = 1:size(theta);
%grad = arrayfun(f, yy);

% Vectorized, after reading this advice:
% https://www.coursera.org/learn/machine-learning/discussions/GVdQ9vTdEeSUBCIAC9QURQ
grad = X' * (H - y) / m;

% =============================================================

end
