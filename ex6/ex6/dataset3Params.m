function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

best_error = 1000000;
best_C = -1;
best_sigma = -1;

%for ix_C = 1:length(C_vec)
%    C = C_vec(ix_C);
%    for ix_sigma = 1:length(sigma_vec)
%        sigma = sigma_vec(ix_sigma);
%        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%        predictions = svmPredict(model, Xval);
%        pred_error = mean(double(predictions ~= yval));
%        if pred_error < best_error
%            fprintf('Found better pred_error %f for C=%f, sigma=%f', pred_error, C, sigma);
%            best_error = pred_error;
%            best_C = C;
%            best_sigma = sigma;
%        end
%    end
%end

C = best_C;
sigma = best_sigma;

C = 1;
sigma = 0.1;
% =========================================================================

end
