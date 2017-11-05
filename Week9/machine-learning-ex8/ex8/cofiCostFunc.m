function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% ------------------------------------------------------------- vectorized implementation of cost function & gradient

% calculating the cost function
error_term = ((X*Theta')-Y).^2;
J = (sum(sum(R.*error_term)))/2;

%regularization cost
Theta_term_cost = (lambda/2)*sum((Theta).^2);
X_term_cost = (lambda/2)*sum((X).^2);
regular_cost = sum(Theta_term_cost+X_term_cost);
J = J + regular_cost;




%calculating the gradient using unrolling 
common_term = (X*Theta')-Y;
user_rated = R.*common_term;  %get only the values where user rated a movie 

X_grad = user_rated * Theta;
Theta_grad = user_rated' * X;

% regularization gradient
X_term_gradient = lambda*X;
Theta_term_gradient = lambda*Theta;

X_grad = X_grad + X_term_gradient;
Theta_grad = Theta_grad + Theta_term_gradient;











% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
