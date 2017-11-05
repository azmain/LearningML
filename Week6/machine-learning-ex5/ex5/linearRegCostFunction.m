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

%--------------------------------------------------------------------- Calculating cost function

hOfTheta = X * theta;
sumOf_cost_function = (sum((hOfTheta-y).^2));
normal_cost_function = sumOf_cost_function / (2*m);

q = sum(theta(2:end,:).^2);
regularized_term = (lambda/(2*m)) * q;

J = normal_cost_function + regularized_term;

%----------------------------------------------------------------------------- Calculating gradient
h = X * theta;
sumOf_gradient = ((h - y)' * X)/m;
regularized_term_gradient = (lambda/m)*[0;theta(2:end)];
grad = sumOf_gradient + regularized_term_gradient' ;






% =========================================================================

grad = grad(:);

end
