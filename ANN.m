clear;
tic;
load ionospheredata.mat; % class & data
%% Pre-processing
% Normalize data, using min-max normalization (from 0 to 1)
for y = 1:size(data, 2)
    minValue = min(data(:, y));
    maxValue = max(data(:, y));
    for x = 1:size(data, 1)
        data(x, y) = ((data(x, y) - minValue) * (1 - 0) / (maxValue - minValue )) + (0);
    end
end
data = data(:, [1 3:end]); % remove NaN attribute

ClassAttribute(find(class == 'g')) = 1;
ClassAttribute(find(class == 'b')) = 0;
% ClassAttribute = ClassAttribute';
class = [ClassAttribute' ~ClassAttribute'];
clear ClassAttribute;
%% Initialize constants
LEARNING_RATE = 0.25;
ACCEPTABLE_ERROR = 0.020;
MAX_EPOCH = 500;
INSTANCE_COUNT = size(data, 1);
ATTRIBUTE_COUNT = size(data, 2);
NEURON_IN_LAYER = [ATTRIBUTE_COUNT; 3; length(unique(class))];
%% Horizontal concatenate attributes with class attributes
data = horzcat(data, class);
clear class;
%% Devide data into 10 boxes
PART = 10;
data = data(randperm(size(data, 1)),:); % shuffle all instances
BoxSize = (size(data, 1) - 1) / PART;
startPart = 1;
endPart = BoxSize;
for part = 1:PART
    DataBox{part} = data(startPart:endPart, :);
    startPart = startPart + 1;
    endPart = endPart + 1;
end
%% Define 10-fold
K_FOLD = 10;
Accuracy = zeros(1, K_FOLD);
bestAccuracy = 0;
bestEpoch = 0;
classOutput = cell(1, K_FOLD);
%%
Epoch = zeros(1, K_FOLD);
AllError = cell(K_FOLD, 1);
for fold = 1:K_FOLD
    %% Initialize weight and bias
    min_rand = -1;
    max_rand = 1;
    input_weight = (max_rand-min_rand) .* rand(NEURON_IN_LAYER(2), NEURON_IN_LAYER(1)) + min_rand;
    hidden_bias = (max_rand-min_rand) .* rand(NEURON_IN_LAYER(2), 1) + min_rand;
    output_bias = (max_rand-min_rand) .* rand(NEURON_IN_LAYER(3), 1) + min_rand;
    min_rand = -sqrt(1.0/ATTRIBUTE_COUNT);
    max_rand = sqrt(1.0/ATTRIBUTE_COUNT);
    hidden_weight = (max_rand-min_rand) .* rand(NEURON_IN_LAYER(3), NEURON_IN_LAYER(2)) + min_rand;
    hidden_sigmoid_sum = zeros(1, NEURON_IN_LAYER(2));
    output_sigmoid_sum = zeros(1, NEURON_IN_LAYER(3));
    %% Divide data set of the fold into test set and training set
%     data = data(randperm(size(data, 1)),:);
    testClass = DataBox{fold}(:, ATTRIBUTE_COUNT+1:end);
    testData = DataBox{fold}(:, 1:ATTRIBUTE_COUNT);
    data = vertcat(DataBox{find(1:size(DataBox, 2) ~= fold)});
    trainingData = data(:, 1:ATTRIBUTE_COUNT);
    trainingClass = data(:, ATTRIBUTE_COUNT+1:end);
    
    TRAINING_DATA_SIZE = size(trainingData, 1);
    TEST_DATA_SIZE = size(testData, 1);
    
    local_gradient = cell(2, 1);
    AverageE = [];
    for epoch = 1:MAX_EPOCH
        E = 0;
        for instance = 1:TRAINING_DATA_SIZE
            %% Feed forward
            layer = 2; % hidden layer
            for node = 1:NEURON_IN_LAYER(layer)
                summation = (trainingData(instance, :) * input_weight(node, :)') + hidden_bias(node);
                sigmoidSummation = Sigmoid(summation);
                hidden_sigmoid_sum(1, node) = sigmoidSummation;
            end
            layer = 3; % output layer
            for node = 1:NEURON_IN_LAYER(layer)
                summation = (hidden_sigmoid_sum(1, :) * hidden_weight(node, :)') + output_bias(node);
                sigmoidSummation = Sigmoid(summation);
                output_sigmoid_sum(1, node) = sigmoidSummation;
                classOutput{1, fold}(1 ,node) = sigmoidSummation;
            end
            %% Local gradient
            layer = 3; % output layer
            for node = 1:NEURON_IN_LAYER(layer)
                output = output_sigmoid_sum(1, node);
                error = trainingClass(instance, node) - output;
                E = E + (error*error);
                local_gradient{layer}(1, node) = error * output * (1 - output);
            end
            layer = 2; % hidden layer
            for node = 1:NEURON_IN_LAYER(layer)
                output = hidden_sigmoid_sum(1, node);
                local_gradient{layer}(1, node) = output * (1 - output) * (local_gradient{layer+1}(1, :) * hidden_weight(:, node));
            end
            %% Adjust new weight & bias
            % Compute delta
            layer = 2; % hidden layer
            for node = 1:NEURON_IN_LAYER(layer)
                % Bias
                delta_bias = LEARNING_RATE * local_gradient{layer}(1, node) * 1;
                hidden_bias(node) = hidden_bias(node) + delta_bias;
                % Weights
                for node_back = 1:NEURON_IN_LAYER(layer-1)
                    delta_weight = LEARNING_RATE * local_gradient{layer}(1, node) * data(instance, node_back);
                    input_weight(node, node_back) = input_weight(node, node_back) + delta_weight;
                end
            end
            layer = 3; % output layer
            for node = 1:NEURON_IN_LAYER(layer)
                % Bias
                delta_bias = (LEARNING_RATE) * local_gradient{layer}(1, node) * 1;
                output_bias(node) = output_bias(node) + delta_bias;
                % Weights
                for node_back = 1:NEURON_IN_LAYER(layer-1)
                    delta_weight = (LEARNING_RATE) * local_gradient{layer}(1, node) * hidden_sigmoid_sum(1, node_back);
                    hidden_weight(node, node_back) = hidden_weight(node, node_back) + delta_weight;
                end
            end
        end
        %% Termination condition
        E = E * 0.5;
        avgE = E / INSTANCE_COUNT; % average E
        if abs(avgE) <= ACCEPTABLE_ERROR
            break
        end
        AllError{fold} = [AllError{fold}; avgE];
    end
%     disp([ num2str(fold) ') average E = ' num2str(AverageE(epoch))]);
    %% Test
    classOutput{fold} = zeros(BoxSize, NEURON_IN_LAYER(3));
    for instance = 1:TEST_DATA_SIZE
        %% Feed forward
        layer = 2; % hidden layer
        for node = 1:NEURON_IN_LAYER(layer)
            summation = (testData(instance, :) * input_weight(node, :)') + hidden_bias(node);
            sigmoidSummation = Sigmoid(summation);
            hidden_sigmoid_sum(1, node) = sigmoidSummation;
        end
        layer = 3; % output layer
        for node = 1:NEURON_IN_LAYER(layer)
            summation = (hidden_sigmoid_sum(1, :) * hidden_weight(node, :)') + output_bias(node);
            sigmoidSummation = Sigmoid(summation);
            output_sigmoid_sum(1, node) = sigmoidSummation;
            classOutput{fold}(instance, node) = sigmoidSummation;
        end
    end
    disp(['FOLD#' num2str(fold) ' epoch = ' num2str(epoch)]);
    correctCount = length(find(round(classOutput{1, fold}(:, 1)) == testClass(:, 1)));
    disp(['correctCount = ' num2str(correctCount)]);
    Accuracy(fold) = correctCount / BoxSize;
    disp(['FOLD#' num2str(fold) ' accuracy = ' num2str(Accuracy(fold))]);
    
    if (Accuracy(fold) > bestAccuracy) && (epoch > bestEpoch)
        bestFold = fold;
        bestEpoch = epoch;
        bestAccuracy = Accuracy(fold);
        save('best_weight_bias.mat', 'input_weight', 'hidden_weight', 'hidden_bias', 'output_bias');
    end
    Epoch(fold) = epoch;
    classLabel(find(classOutput{1, fold}(:,1) >= classOutput{1, fold}(:, 2)), 1) = 'g';
    classLabel(find(classOutput{1, fold}(:,1) < classOutput{1, fold}(:, 2)), 1) = 'b';
end

load('best_weight_bias.mat', 'input_weight', 'hidden_weight', 'hidden_bias', 'output_bias');
PRINT_INPUT_WEIGHT = input_weight
PRINT_HIDDEN_WEIGHT = hidden_weight
PRINT_HIDDEN_BIAS = hidden_bias
PRINT_OUTPUT_BIAS = output_bias
% PRINT_AVERAGEE = AverageE'
PRINT_ACCURACY = Accuracy'
PRINT_EPOCH = Epoch'

averageErrorFold = [];
for a = 1:10
    epochError = AllError{a, 1};
    averageErrorFold = [averageErrorFold; sum(epochError) / size(epochError, 1)];
end
averageErrorFold

disp(['Average Accuracy of 10 folds = ' num2str(sum(Accuracy) / length(Accuracy))])

toc