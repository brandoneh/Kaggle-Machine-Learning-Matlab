% This script was used to train a binary classifier using data from a Kaggle competition.
% Training and testing data consisted of GIST and CNN image data of a
% scene, in which the classifier had to ascertain whether a live object was
% present. This was accompanied by additional GIST and CNN data containing
% empty cells that had to be imputed, data detailing the confidence with
% which annotators assigned classes to images and the proportion of
% positive and negative classes within the final test data. This data was
% used to train a Gaussian Support Vector Machine using the Classification
% Learner.

%

importScript; % Imports training and testing data

inputData_wEmpty = [inputData; emptyData]; % Concatenates the input data with the empty data
inputData_wEmptyAnnotated = [inputData_wEmpty,confidenceData(:,2)];% Adds the confidence annotations

%% Filters the annotations

[R,C] = size(inputData_wEmptyAnnotated);
inputData_wEmptyFiltered = zeros(1,C); % Initializes a matrix for filtering by annotations

% Conditional for loop that adds samples with a confidence of 1 to a new
% matrix

for bb = 1:R
   if(inputData_wEmptyAnnotated(bb,4611) == 1) 
       inputData_wEmptyFiltered = [inputData_wEmptyFiltered;inputData_wEmptyAnnotated(bb,:)]; 
   end
end

% Deletes the first row as it is empty
inputData_wEmptyFiltered(1,:) = [];

% Randomly sorts the rows before removing samples in the proportion step
[filteredR, filteredC] = size(inputData_wEmptyFiltered);
inputData_wEmptyFiltered = inputData_wEmptyFiltered(randperm(filteredR),:);
trainData = inputData_wEmptyFiltered;

% Impute the Median
trainDataMedian = Imputer(trainData,'median');

%% Proportion for training

% Finds the size of the current dataset
trainMSize = size(trainDataMedian);

trainingPos = 0;
trainingNeg = 0;

 % checks the number of positive and negative samples in the data set
for bb = 1:trainMSize(1)
   if(trainDataMedian(bb,4610) == 0)
      trainingNeg = trainingNeg + 1;
   else
      trainingPos = trainingPos + 1;
   end
end

% Calculates the number of positive and negative samples needed
wantedTrainingNeg = trainingNeg;
wantedTrainingPos = round((trainingNeg * 1/0.5714) * 0.4286);


% Initializes two matrics for storing negative and positive values.
trainDataMedianPos = zeros(1,trainMSize(2));
trainDataMedianNeg = zeros(1,trainMSize(2));

% Stores all positive samples
for pp = 1:trainMSize(1)
    if(trainDataMedian(pp,4610) == 1)
        trainDataMedianPos = [trainDataMedianPos;trainDataMedian(pp,:)];
    end
end
trainDataMedianPos(1,:) = [];

% Stores all negative samples
for oo = 1:trainMSize(1)
    if(trainDataMedian(oo,4610) == 0)
        trainDataMedianNeg = [trainDataMedianNeg;trainDataMedian(oo,:)];
    end
end
trainDataMedianNeg(1,:) = [];

% Creates a matrix with the correct number of samples from each class
trainDataMedianProp = [trainDataMedianPos(1:wantedTrainingPos,:);trainDataMedianNeg(1:wantedTrainingNeg,:)];
trainDataMedianPropSize = size(trainDataMedianProp);

% Shuffles the data to avoid it being sorted by group
trainDataMedianProp = trainDataMedianProp(randperm(trainDataMedianPropSize(1)),:);

finalTrainingData = trainDataMedianProp(:,2:end-1);

%The classifier app is then used to produce a medium gaussian SVM with
%kernel size 68 and box constraint 1.

TrainedClassifier = trainClassifier(finalTrainingData);


