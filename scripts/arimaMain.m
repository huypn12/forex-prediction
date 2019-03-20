%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cfg = config();

[eurusd, featureNames] = eurusdDataset(cfg.dataset.csvPath, "");
[eurusdTrain, ~, eurusdTest] = eurusdPartition(...
    eurusd, cfg.dataset.trainSetRatio);

%% Loop through Open-High-Low-Close univariate timeseries
for i = 1:4
    YTrain = eurusdTrain(:, i);
    YTest = eurusdTest(:, i);
    
    if cfg.execMode == "rerun"
        [estMdl, YPred, aic, rmse, mape] = appArimaRerun(...
            YTrain, YTest, featureNames(i));
    else
        params = cfg.arima.params(i, :);
        [estMdl, YPred, aic, rmse, mape] = appArimaVerify(...
            YTrain, YTest, featureNames(i), params{:});
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [estMdl, YPred, aic, rmse, mape] = appArimaVerify(...
        YTrain, YTest, featureName, p, d, q)
    %% Verify the parameter set on test data
    mdl = arima(p,d,q);
    estMdl = estimate(mdl, YTrain);
    mdlSummary = summarize(estMdl);
    aic = mdlSummary.AIC;
    YPred = arimaPredict(estMdl, YTest, 1);
    [rmse, mape] = computePredError(YTest, YPred);
    visualizeResult(estMdl, YPred, YTest, rmse, mape, featureName);
end

function [estMdl, YPred, aic, rmse, mape] = appArimaRerun(...
        YTrain, YTest, featureName)
    %% Run from beginning
    plotAcfPacf(YTrain, featureName);
    estMdl = arimaParamsSearch(YTrain, 1, 1, 1);
    mdlSummary = summarize(estMdl);
    aic = mdlSummary.AIC;
    YPred = arimaPredict(mdl, YTest, 1);
    [rmse, mape] = computePredError(YTest, YPred);
    visualizeResult(estMdl, YPred, YTest, rmse, mape);
    resultFilename = strcat('arima_', featureName);
    save(resultFilename, 'estMdl', 'YPred', 'YTest', 'rmse', 'mape');
end

function plotAcfPacf(Y, featureName)
    %% Plot ACF and PCAF for univariate timeseries
    Y = diff(log(Y));
    figureTag = strcat("ACF and PACF of ", featureName);
    figure('Name', figureTag);
    subplot(2,1,1)
    autocorr(Y)
    title("ACF")
    subplot(2,1,2)
    parcorr(Y)
    title("PACF")
end


function model = arimaParamsSearch(YTrain, maxP, maxD, maxQ)
    %% Grid search to confirm (p, d, q) parameters for ARIMA
    minAic = Inf;
    resultModel = arima(1,0,1);
    for p = 1:maxP
        for d = 0:maxD
            for q = 1:maxQ
                mdl = arima(p,d,q);
                try
                    [estMdl, ~, logL, info] = estimate(mdl, YTrain);
                catch
                    warning("Unstable estimation");
                    continue;
                end
                mdlSum = summarize(estMdl);
                aic = mdlSum.AIC;
                if aic < minAic
                    minAic = aic;
                    resultModel = estMdl;
                end
            end
        end
    end
    model = resultModel;
end

function YPred = arimaPredict(estMdl, YTest, numResponse)
    %% Predict num_response values ahead from model and Y
    YTest = YTest.';
    YPred = zeros(1, size(YTest, 2));
    lags = max(estMdl.P, estMdl.Q);
    YPred(1:lags) = YTest(1);
    i = 1;
    while i <= (size(YTest, 2) - lags)
        observations = YTest(i:(i+lags-1));
        [tmp, ~] = forecast(estMdl, numResponse, 'Y0', observations.');
        YPred((i+lags):(i+lags+numResponse-1)) = tmp;
        i = i + numResponse;
    end
    YPred = YPred.';
end

function [errRmse, errMape] = computePredError(YTest, YPred)
    errRmse = rmse(YTest, YPred);
    errMape = mape(YTest, YPred);
end


function visualizeResult(estMdl, YPred, YTest, rmse, mape, featureName)
    %% Plot the predicted value, observed value, and error
    modelDesc = strcat("ARIMA(", num2str(estMdl.P),...
        ",", num2str(estMdl.D),...
        ",", num2str(estMdl.Q));
    figureTag = strcat(modelDesc + " on EURUSD BID, ", featureName, " price");
    figure('Name', figureTag);
    plot(YTest)
    hold on
    plot(YPred,'.-')
    hold off
    legend(["Observed" "Predicted"])
    ylabel("EURUSD")
    title("RMSE=" + rmse + " MAPE=" + mape);
end
