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

z = X*theta;
h = sigmoid(z);
theta_reg = theta(2:size(theta));
theta_reg_sq = theta_reg.^2;
theta_reg_sq_sum = sum(theta_reg_sq);

cost_matrix = y.*log(h) + (1-y).*log(1-h);
reg_term = (lambda/(2*m))*theta_reg_sq_sum;
J = (-1/m)*sum(cost_matrix(:)) + reg_term;


theta_reg_grad = [0;theta_reg];
grad = (1/m)*(X'*(h-y)) + (lambda/m).*theta_reg_grad;

 
% =============================================================

end
