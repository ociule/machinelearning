function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

H = sigmoid(X * theta);
J = sum(-y .* log(H) - (1 - y) .* log(1 - H)) / m;

% See https://www.coursera.org/learn/machine-learning/module/mgpv7/discussions/0DKoqvTgEeS16yIACyoj1Q
theta(1) = 0;
J += lambda * theta' * theta / (2 * m);

% First try: unvectorized
%grad(1) = 1/m * sum((sigmoid(X * theta) - y) .* X(:, 1));

%for i = 2:size(theta)
%    grad(i) = 1/m * sum((sigmoid(X * theta) - y) .* X(:, i)) + lambda / m * theta(i);
%end

% Second try, with arrayfun
% selector = ones(size(theta), 1);
% selector(1) = 0;
% f = @(i) 1/m * sum((sigmoid(X * theta) - y) .* X(:, i)) + selector(i) * lambda / m * theta(i);

%yy = 1:size(theta);
%grad = arrayfun(f, yy);

% Vectorized, after reading this advice:
% https://www.coursera.org/learn/machine-learning/discussions/GVdQ9vTdEeSUBCIAC9QURQ

% Remember that theta(1) = 0;
grad = X' * (H - y) / m + theta * lambda / m;


% =============================================================

end
