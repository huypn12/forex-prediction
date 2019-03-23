MAIN();

function MAIN()
    cfg = config();
    
    [eurusd, ~] = eurusdDataset(cfg.dataset.csvPath, "");
    eurusdStandardized = eurusdStandardize(eurusd);
    [eurusdTrain, ~, YTest] = eurusdPartition(...
        eurusdStandardized, cfg.dataset.trainSetRatio);

    if cfg.execMode == "verify"
        [lstmOpenModel, lstmHighModel, lstmLowModel, lstmCloseModel] =...
            loadTrainedModels(cfg.lstm.multi.savedModelsFile);
        verifyLstmModels(lstmOpenModel, lstmHighModel, lstmLowModel, lstmCloseModel, YTest);
        return;
    end
    
    [XTrain, YTrain] = prepareTrainData(eurusdTrain, cfg.numLags);
    [lstmOpenModel, lstmHighModel, lstmLowModel, lstmCloseModel] = ...
        trainLstmModels(XTrain, YTrain, 8, 1);
    saveTrainedModels(cfg.lstm.multi.savedModelsFile,...
        lstmOpenModel, lstmHighModel, lstmLowModel, lstmCloseModel);
    verifyLstmModels(lstmOpenModel, lstmHighModel, lstmLowModel, lstmCloseModel, YTest);
    
end

function [lstmOpenModel, lstmHighModel, lstmLowModel, lstmCloseModel] =...
        loadTrainedModels(modelFile)
    load(modelFile,... 
        'lstmOpenModel', 'lstmHighModel', 'lstmLowModel', 'lstmCloseModel');
end

function saveTrainedModels(modelFile, lstmOpenModel, lstmHighModel, lstmLowModel, lstmCloseModel)
    save(modelFile,...
        'lstmOpenModel', 'lstmHighModel', 'lstmLowModel', 'lstmCloseModel');
end

function verifyLstmModels(lstmOpenModel, lstmHighModel, lstmLowModel, lstmCloseModel, YTest)
    YOpenTest = YTest(:, 1);
    [YOpenPred, openRmse, openMape] = predictLstmModel(lstmOpenModel, YTest);
    visualizeResult(YOpenTest, YOpenPred, openRmse, openMape, 'Open');
    YHighTest = YTest(:, 2);
    [YHighPred, highRmse, highMape] = predictLstmModel(lstmHighModel, YTest);
    visualizeResult(YHighTest, YHighPred, highRmse, highMape, 'High');
    YLowTest = YTest(:, 3);
    [YLowPred, lowRmse, lowMape] = predictLstmModel(lstmLowModel, YTest);
    visualizeResult(YLowTest, YLowPred, lowRmse, lowMape, 'Low');
    YCloseTest = YTest(:, 4);
    [YClosePred, closeRmse, closeMape] = predictLstmModel(lstmCloseModel, YTest);
    visualizeResult(YCloseTest, YClosePred, closeRmse, closeMape, 'Close');
    finalRmse = sqrt(0.25 * (openRmse^2 + highRmse^2 + lowRmse^2 + closeRmse^2));
    finalMape = 0.25 * (openMape + highMape + lowMape + closeMape);
    fprintf('Overall error: RMSE=%f MAPE=%f', finalRmse, finalMape);
end

function [lstmOpenModel, lstmHighModel, lstmLowModel, lstmCloseModel] = ...
        trainLstmModels(XTrain, YTrain, numFeatures, numResponses)
    YOpenTrain = YTrain(:, 1);
    lstmOpenModel = buildAndTrainLstmModel(XTrain, YOpenTrain, numFeatures, numResponses);
    YHighTrain = YTrain(:, 2);
    lstmHighModel = buildAndTrainLstmModel(XTrain, YHighTrain, numFeatures, numResponses);    
    YLowTrain = YTrain(:, 3);
    lstmLowModel = buildAndTrainLstmModel(XTrain, YLowTrain, numFeatures, numResponses);
    YCloseTrain = YTrain(:, 4);
    lstmCloseModel = buildAndTrainLstmModel(XTrain, YCloseTrain, numFeatures, numResponses);
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

function [lstmModel, options] = buildAndTrainLstmModel(...
        XTrain, YTrain, numFeatures, numResponses)
    %% Train a model for one series of observation
    numHiddenUnits = 125;
    lstmModel = [ ...
        sequenceInputLayer(numFeatures),...
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence'),...
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence'),...
        fullyConnectedLayer(numHiddenUnits),...
        dropoutLayer(0.1),...
        lstmLayer(numHiddenUnits, 'OutputMode', 'last'),...
        fullyConnectedLayer(numResponses),...
        regressionLayer
        ];
    
    options = trainingOptions(...
        'adam', ...
        'MaxEpochs', 3, ... 
        'MiniBatchSize', 32,...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress'...
        );
    
    lstmModel = trainNetwork(XTrain, YTrain, lstmModel, options);
end

function [YPred, rmse, mape] = predictLstmModel(lstmModel, YTest)
    YTest = YTest.';
    YPred = [];
    numTimeSteps = size(YTest, 2);
    for i = 1:numTimeSteps
        [lstmModel, YPred(i)] = predictAndUpdateState(...
            lstmModel,YTest(:,i),'ExecutionEnvironment','cpu');
    end
    [rmse, mape] = computePredError(YTest, YPred);
end

function [errRmse, errMape] = computePredError(YTest, YPred)
    errRmse = rmse(YTest, YPred);
    errMape = mape(YTest, YPred);
end

function visualizeResult(YPred, YTest, rmse, mape, featureName)
    %% Plot the predicted value, observed value, and error
    modelDesc = "LSTM, 8 features multivariate";
    figureTag = strcat(modelDesc + ", EURUSD BID, ", featureName, " price");
    figure('Name', figureTag);
    plot(YTest, 'b')
    hold on
    plot(YPred, 'r')
    hold off
    legend(["Observed" "Predicted"])
    ylabel("EURUSD")
    title("RMSE=" + rmse + " MAPE=" + mape);
end

