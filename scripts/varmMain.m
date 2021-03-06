MAIN();

function MAIN()
    cfg = config();
    
    [eurusd, featureNames] = eurusdDataset(cfg.dataset.csvPath, "");
    [YTrain, ~, YTest] = eurusdPartition(...
        eurusd, cfg.dataset.trainSetRatio);
    YTrain = YTrain(:, 1:4);
    YTest = YTest(:, 1:4);
    
    if cfg.execMode == "verify"    
        load(cfg.varm.savedModelsFile, 'varmMdl');
        [YPred, varmRmse, varmMape] = predictVarm(varmMdl, YTest, cfg.numResponses);
        mesg = strcat('Optimal parameter P=', num2str(varmMdl.P),...
            ' RMSE=', num2str(varmRmse), ' MAPE=', num2str(varmMape));
        display(mesg);
        visualizeResult(varmMdl, YPred, YTest, featureNames);
        return
    end
    
    [varmMdl, ~] = searchVarmParams(YTrain, cfg.numLags);
    save(cfg.varm.savedModelsFile, 'varmMdl');
    [YPred, varmRmse, varmMape] = predictVarm(varmMdl, YTest, cfg.numResponses);
    mesg = strcat('Optimal parameter P=', num2str(varmMdl.P),...
        ' RMSE=', num2str(varmRmse), ' MAPE=', num2str(varmMape));
    display(mesg);
    visualizeResult(varmMdl, YPred, YTest, featureNames);
    
end

function [varmMdl, aic] = searchVarmParams(YTrain, numLags)
    dimInput = size(YTrain, 2);
    minAic = Inf;
    minMdl = varm(dimInput, 1);
    for lag = 1:numLags
        mdl = varm(dimInput, lag);
        estMdl = estimate(mdl, YTrain);
        results = summarize(estMdl);
        aic = results.AIC;
        if (aic < minAic)
            minAic = aic;
            minMdl = estMdl;
        end
    end
    varmMdl = minMdl;
    aic = minAic;
end

function [YPred, mape, rmse] = predictVarm(estMdl, YTest, numResponse)
    %% Predict num_response values ahead from model and Y
    YTest = YTest.';
    YPred = zeros(size(YTest));
    lags = estMdl.P;
    YPred(:, 1:lags) = YTest(1);
    i = 1;
    while i <= (size(YTest, 2) - lags)
        observations = YTest(:, i:(i+lags-1));
        [tmp, ~] = forecast(estMdl, numResponse, observations.');
        YPred(:,(i+lags):(i+lags+numResponse-1)) = tmp;
        i = i + numResponse;
    end
    [rmse, mape] = computePredError(YTest, YPred);
    YPred = YPred.';
end


function [errRmse, errMape] = computePredError(YTest, YPred)
    errRmse = rmse(YTest, YPred);
    errMape = mape(YTest, YPred);
end

function visualizeResult(estMdl, YTest, YPred, featureNames)
    %% Plot the predicted value, observed value, and error
    modelSummary = summarize(estMdl);
    modelDesc = strcat("VAR(", num2str(estMdl.P), ") AIC=",...
        num2str(modelSummary.AIC));
    for i = 1:4 
        uniRmse = rmse(YTest(:, i), YPred(:, i));
        uniMape = mape(YTest(:, i), YPred(:, i));
        figureTag = strcat(modelDesc, ", EURUSD BID, ", featureNames(i));
        figure('Name', figureTag)
        plot(YTest(:, i), 'b')
        hold on
        plot(YPred(:, i), 'r')
        hold off
        legend(["Observed" "Predicted"])
        ylabel("EURUSD BID price")
        title(featureNames(i) + ": RMSE=" + uniRmse + " MAPE=" + uniMape);
    end
end

