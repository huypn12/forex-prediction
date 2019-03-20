%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cfg = config();

[eurusd, featureNames] = eurusdDataset(cfg.dataset.csvPath, "");
eurusdStandardized = eurusdStandardize(eurusd);
[eurusdTrain, YValid, YTest] = eurusdPartition(...
    eurusdStandardized, cfg.dataset.trainSetRatio);
[XTrain, YTrain] = prepareTrainData(eurusdTrain, cfg.maxLags);

YOpenTrain = YTrain(:, 1);
[lstmNet, options] = lstmMultivariate(size(XTrain, 1), 1);
lstmNet = trainNetwork(XTrain, YOpenTrain, lstmNet, options);
x = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [YPred, YTest, rmse, mape] = appLstmMultivariateVerify(YTest, cfg)
end

function [lstmNet, YPred, YTest, rmse, mape] = appLstmMultivariateTrain(...
        XTrain, YTrain, YTest, cfg)

end



function [XTrain, YTrain] = prepareTrainData(trainset, k)
    %% Fold the original dataset into chunks of size lag
    chunkCount = ceil(size(trainset, 1) / k);
    XTrain = {};%zeros(chunkCount, size(trainset, 2), k);
    YTrain = zeros(chunkCount, size(trainset, 2));
    for i = 1:(size(trainset, 1) - k)
        tmpX = trainset(i:(i + k - 1), :);
        tmpY = trainset((i + k), :);
        XTrain{i} = tmpX.';
        YTrain(i, :) = tmpY;
    end
    XTrain = XTrain.';
end



function [lstmNetwork, options] = lstmMultivariate(numFeatures, numResponses)
    numHiddenUnits = 50;
    lstmNetwork = [ ...
        sequenceInputLayer(numFeatures),...
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence'),...
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence'),...
        fullyConnectedLayer(numHiddenUnits),...
        dropoutLayer(0.5),...
        fullyConnectedLayer(numHiddenUnits),...
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence'),...
        lstmLayer(numHiddenUnits, 'OutputMode', 'last'),...
        fullyConnectedLayer(numResponses),...
        regressionLayer
        ];
    
    options = trainingOptions(...
        'adam', ...
        'MaxEpochs', 5, ... %% For test only
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress'...
        );
    
end