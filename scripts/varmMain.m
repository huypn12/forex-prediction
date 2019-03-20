%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cfg = config();

[eurusd, featureNames] = eurusdDataset(cfg.dataset.csvPath, "");
[YTrain, ~, YTest] = eurusdPartition(...
    eurusd, cfg.dataset.trainSetRatio);
YTrain = YTrain(:, 1:4);
YTest = YTest(:, 1:4);

if cfg.execMode == "train"
    [estMdl, YPred, aic, rmse, mape] = appVarmTrain(...
        YTrain, YTest, featureNames, cfg);
else
    p = cfg.var.P;
    [estMdl, YPred, aic, rmse, mape] = appVarmVerify(YTrain, YTest, P);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [varmMdl, YPred, aic, rmse, mape] = appVarmTrain(...
        YTrain, YTest, featureNames, cfg)
    [varmMdl, aic] = varmSearchParams(YTrain, cfg.maxLags);
    [YPred, rmse, mape] = varmPredict(varmMdl, YTest, 1);
    save(modelFilename(), 'varmMdl');
    visualizeResult(varmMdl, YPred, YTest, rmse, mape, featureNames);
end

function filename = modelFilename()
    filename = strcat('varma_ohlc_', datestr(now,'yyyymmddTHHMMSS'));
end

function [YPred, aic, rmse, mape] = appVarmVerify(YTest, cfg)
    load(cfg.varm.savedModel, 'varmMdl');
    varmMdlSummary = summarize(varmMdl);
    aic = varmMdlSummary.aic;
    [YPred, rmse, mape] = varmPredict(varmMdl, YTest, 1);
end

function [varmMdl, aic] = varmSearchParams(YTrain, maxLags)
    dimInput = size(YTrain, 2);
    minAic = Inf;
    minMdl = varm(dimInput, 1);
    for lag = 1:maxLags
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
end

function [YPred, mape, rmse] = varmPredict(estMdl, YTest, numResponse)
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

function visualizeResult(estMdl, YTest, YPred, mapeErr, rmseErr, featureNames)
    %% Plot the predicted value, observed value, and error
    modelDesc = strcat("VAR(", num2str(estMdl.P), ")");
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
        ylabel("EURUSD")
        title(featureNames(i) + ": RMSE=" + uniRmse + " MAPE=" + uniMape);
    end
end

