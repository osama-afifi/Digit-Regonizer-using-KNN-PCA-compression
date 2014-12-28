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
%fprintf('.................... Phase 1 .......................\n')
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
displayData(X(sel, :));

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ================ Part 2: Principle Component Analysis ================
%  Running PCA and display it's eigenvectors
%fprintf('.................... Phase 2 .......................\n')

fprintf('\nRunning PCA on example dataset.\n\n');
%  Before running PCA, it is important to first normalize X
X_norm = X;
%[X_norm, mu, sigma] = featureNormalize(X);
%  Run PCA
[U, S] = pca(X_norm);

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% =================== Part 3: Dimension Reduction ===================
%  map the data onto the first k eigenvectors. 

%fprintf('.................... Phase 3 .......................\n')
fprintf('\nDimension reduction on example dataset.');

%  Project the data onto K dimensions for PCA
%K = 50;
for K = 5:50;
    
fprintf('\nProject the data onto %d dimensions for PCA', K);


Z = projectData(X_norm, U, K);
%fprintf('Program paused. Press enter to continue.\n');
%pause;


%% ================= Part 4: Predict using KNN =================
%  we would like to use it to predict the labels of the training set using KNN

%fprintf('.................... Phase 4 .......................\n');
k = 15;%input(' Please Enter the number of nearest neighbours you want to classify on: ');
%fprintf('\nTrying on a sample data\n', Z(1));
classes = 10;
% Trying on a specific Sample
sample_no = 101;%input(' Please Enter the sample number you want to classify: ');
sample_data = Z(sample_no,:);
pred = KNN(sample_data,Z,y, k, classes);
pred(pred==10) = 0;
%fprintf('\nPredicted Digit: %d\n', pred);
supposedVal = y(sample_no);
supposedVal(supposedVal==10) = 0;
%fprintf('\nSupposed Digit: %d\n', y(sample_no));
%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ================= Part 5: Calculating Accuracy =================

%fprintf('.................... Phase 5 .......................\n')
fprintf('\nCalculating Total Training Accuracy...');


sampleNumber=100;
classVec = zeros(sampleNumber,1);
labelVec = y(1:sampleNumber,:);
reverseStr = '';

for idx = 1:sampleNumber
    
    classVec(idx) = KNN(Z(idx,:),Z,y, k, classes);  
   % Display the progress
   percentDone = 100 * idx / sampleNumber;
   msg = sprintf('Percentage done: %3.1f%', percentDone); %Don't forget this semicolon
   fprintf([reverseStr, msg]);
   reverseStr = repmat(sprintf('\b'), 1, length(msg));
   
end;

fprintf('\nTraining Set Accuracy: %f\n', mean(double(classVec == labelVec)) * 100.0);

%fprintf('Program paused. Press enter to continue.\n');
%pause;

end;
