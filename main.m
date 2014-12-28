%% 29/12/2014
%% Digit Recognition Labelling using K-nearest Neighbour

%  Procedure
%  ------------
% 
%	1  Load and Manipulate the Data.
%	2.1  Normalize the Data
%	2.2  Run the PCA
%   2.3  Plot the Eigenvectors
%   3    Map the Data on the new K-Dimensions
%	4    Run KNN on a test sample the PCA Data
%	5	 Calculate Training Accuracy (Optional)


%% Initialization
clear ; close all; clc

					  
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%  We also normalize the data

% Load Training Data
fprintf('.................... Phase 1 .......................\n')
fprintf('Loading Data File ...\n')
Data = load('Data/train.csv');
fprintf('Setting up Label Vector ...\n')
y = Data(:,1);
y( y==0 )= 10; % Mapping 0 into 10
fprintf('Setting up Feature Matrix ...\n')
feature_columns = [2 : size(Data,2)];
X = Data(:,feature_columns);
size(X,1)
size(X,2)
m = size(X, 1);
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
fprintf('Visualize Data ...\n')
%displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Principle Component Analysis ================
%  Running PCA and display it's eigenvectors
fprintf('.................... Phase 2 .......................\n')

fprintf('\nRunning PCA on example dataset.\n\n');
%  Before running PCA, it is important to first normalize X
X_norm = X;
%[X_norm, mu, sigma] = featureNormalize(X);
%  Run PCA
[U, S] = pca(X_norm);

%  Draw the eigenvectors centered at mean of data. These lines show the
%  directions of maximum variations in the dataset.
%hold on;
%drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
%drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
%hold off;

fprintf('Top eigenvector: \n');
fprintf(' U(:,1) = %f %f \n', U(1,1), U(2,1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Dimension Reduction ===================
%  map the data onto the first k eigenvectors. 

fprintf('.................... Phase 3 .......................\n')
fprintf('\nDimension reduction on example dataset.\n\n');

%  Plot the normalized dataset (returned from pca)
%plot(X_norm(:, 1), X_norm(:, 2), 'bo');
%axis([-4 3 -4 3]); axis square

%  Project the data onto K dimensions
K = 5;
Z = projectData(X_norm, U, K);
fprintf('Projection of the first example: %f\n', Z(1));

%X_rec  = recoverData(Z, U, K);
%fprintf('Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));

%  Draw lines connecting the projected points to the original points
%hold on;
%plot(X_rec(:, 1), X_rec(:, 2), 'ro');
%for i = 1:size(X_norm, 1)
%    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
%end
%hold off

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 4: Predict using KNN =================
%  we would like to use it to predict the labels of the training set using KNN

fprintf('.................... Phase 4 .......................\n')
sample_no = 5;
sample_data = Z(sample_no,:);
k = 10;
classes = 10;
pred = KNN(sample_data,Z, k, classes);
pred(pred==10) = 0;
fprintf('\nPredicted Digit: %d\n', pred);
supposedVal = y(sample_no);
supposedVal(supposedVal==10) = 0;
fprintf('\nSupposed Digit: %d\n', y(sample_no));


%% ================= Part 5: Calculating Accuracy =================

%fprintf('.................... Phase 5 .......................\n')
%k = 10;
%classes = 10;
%pred = KNN(X, sample_data, k, classes);
%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

