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

J = sum(-y .* log(sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta))) / m;

for i = 2:size(theta)
  J += lambda * theta(i) ^ 2 / (2 * m);
end

% grad = sum(sigmoid(X * theta) - y) .* X / m;


selector = ones(size(theta), 1);
selector(1) = 0;
f = @(i) 1/m * sum((sigmoid(X * theta) - y) .* X(:, i)) + selector(i) * lambda / m * theta(i);

yy = 1:size(theta);
grad = arrayfun(f, yy);

%grad(1) = 1/m * sum((sigmoid(X * theta) - y) .* X(:, 1));

%for i = 2:size(theta)
%    grad(i) = 1/m * sum((sigmoid(X * theta) - y) .* X(:, i)) + lambda / m * theta(i);
%end



% =============================================================

end
