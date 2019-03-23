MAIN();

function MAIN()
    cfg = config();
    cfg.numLags = 1;
    [eurusd, featureNames] = eurusdDataset(cfg.dataset.csvPath, "");
    [YTrain, ~, YTest] = eurusdPartition(... 
        eurusd, cfg.dataset.trainSetRatio);
    
    if cfg.execMode == "verify"
        [arimaOpenMdl, arimaHighMdl, arimaLowMdl, arimaCloseMdl] =...
            loadArimaModels(cfg.arima.savedModelsFile);
        verifyArimaModels(...
            arimaOpenMdl, arimaHighMdl, arimaLowMdl, arimaCloseMdl, YTest);
        return;
    end
    
    [arimaOpenMdl, arimaHighMdl, arimaLowMdl, arimaCloseMdl] = ...
        trainArimaModels(YTrain, cfg.numLags, featureNames);
    saveArimaModels(cfg.arima.savedModelsFile,...
        arimaOpenMdl, arimaHighMdl, arimaLowMdl, arimaCloseMdl);
    verifyArimaModels(...
        arimaOpenMdl, arimaHighMdl, arimaLowMdl, arimaCloseMdl, YTest);
end

function saveArimaModels(...
    modelFile, arimaOpenMdl, arimaHighMdl, arimaLowMdl, arimaCloseMdl)
save(modelFile,...
    'arimaOpenMdl', 'arimaHighMdl', 'arimaLowMdl', 'arimaCloseMdl');
end

function [arimaOpenMdl, arimaHighMdl, arimaLowMdl, arimaCloseMdl] = loadArimaModels(modelFile)
    load(modelFile,...
        'arimaOpenMdl', 'arimaHighMdl', 'arimaLowMdl', 'arimaCloseMdl');
end

function verifyArimaModels(arimaOpenMdl, arimaHighMdl, arimaLowMdl, arimaCloseMdl, YTest)
    YOpenTest = YTest(:, 1);
    [YOpenPred, openRmse, openMape] = arimaPredict(arimaOpenMdl, YOpenTest, 1);
    visualizeResult(arimaOpenMdl, YOpenPred, YOpenTest, 'open');    
    YHighTest = YTest(:, 2);
    [YHighPred, highRmse, highMape] = arimaPredict(arimaHighMdl, YHighTest, 1);
    visualizeResult(arimaHighMdl, YHighPred, YHighTest, 'high');
    YLowTest = YTest(:, 3);
    [YLowPred, lowRmse, lowMape] = arimaPredict(arimaLowMdl, YLowTest, 1);
    visualizeResult(arimaLowMdl, YLowPred, YLowTest, 'low');
    YCloseTest = YTest(:, 4);
    [YClosePred, closeRmse, closeMape] = arimaPredict(arimaCloseMdl, YCloseTest, 1);
    visualizeResult(arimaCloseMdl, YClosePred, YCloseTest, 'close');
    finalRmse = sqrt(0.25 * (openRmse^2 + highRmse^2 + lowRmse^2 + closeRmse^2));
    finalMape = 0.25 * (openMape + highMape + lowMape + closeMape);
    fprintf('Overall error: RMSE=%f MAPE=%f', finalRmse, finalMape);
end

function [arimaOpenMdl, arimaHighMdl, arimaLowMdl, arimaCloseMdl] =...
    trainArimaModels(YTrain, numLags, featureNames)    
    % Visualize ACF and PACF of Open-High-Low-Close timeseries
    for i = 1:4
        plotAcfPacf(YTrain(:, i), featureNames(i));
    end
    YOpenTrain = YTrain(:, 1);
    [arimaOpenMdl, openAic] = arimaParamsSearch(YOpenTrain, numLags, 1, numLags);
    YHighTrain = YTrain(:, 2);
    [arimaHighMdl, highAic] = arimaParamsSearch(YHighTrain, numLags, 1, numLags);
    YLowTrain = YTrain(:, 3);
    [arimaLowMdl, lowAic] = arimaParamsSearch(YLowTrain, numLags, 1, numLags);
    YCloseTrain = YTrain(:, 4);
    [arimaCloseMdl, closeAic] = arimaParamsSearch(YCloseTrain, numLags, 1, numLags);
    
    fprintf('Optimal parameters for Low (p=%d,d=%d,q=%d) AIC=%f\n',...
        arimaLowMdl.P, arimaLowMdl.D, arimaLowMdl.Q, openAic);
    fprintf('Optimal parameters for High (p=%d,d=%d,q=%d) AIC=%f\n',...
        arimaHighMdl.P, arimaHighMdl.D, arimaHighMdl.Q, highAic);
    fprintf('Optimal parameters for Open (p=%d,d=%d,q=%d) AIC=%f\n',...
        arimaOpenMdl.P, arimaOpenMdl.D, arimaOpenMdl.Q, lowAic);
    fprintf('Optimal parameters for Close (p=%d,d=%d,q=%d) AIC=%f\n',...
        arimaCloseMdl.P, arimaCloseMdl.D, arimaCloseMdl.Q, closeAic);
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


function [model, aic] = arimaParamsSearch(YTrain, maxP, maxD, maxQ)
    %% Grid search to confirm (p, d, q) parameters for ARIMA
    minAic = Inf;
    resultModel = arima(1,0,1);
    for p = 1:maxP
        for d = 0:maxD
            for q = 1:maxQ
                mdl = arima(p,d,q);
                try
                    estMdl = estimate(mdl, YTrain);
                catch
                    warning("The model's coefficients cannot be estimated due to numerical instability.");
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
    aic = minAic;
end


function [errRmse, errMape] = computePredError(YTest, YPred)
    errRmse = rmse(YTest, YPred);
    errMape = mape(YTest, YPred);
end


function visualizeResult(estMdl, YPred, YTest, featureName)
    %% Plot the predicted value, observed value, and error
    [rmse, mape] = computePredError(YPred, YTest);
    modelSummary = summarize(estMdl);
    modelDesc = strcat("ARIMA(", num2str(estMdl.P),...
        ",", num2str(estMdl.D),...
        ",", num2str(estMdl.Q), ')',...
        " AIC=", num2str(modelSummary.AIC));
    figureTag = strcat(modelDesc + ", EURUSD BID ", featureName, " price");
    figure('Name', figureTag);
    plot(YTest, 'b')
    hold on
    plot(YPred, 'r')
    hold off
    legend(["Observed" "Predicted"])
    ylabel("EURUSD BID price")
    title("RMSE=" + rmse + " MAPE=" + mape);
end


function [YPred, rmse, mape] = arimaPredict(estMdl, YTest, numResponse)
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
    [rmse, mape] = computePredError(YTest, YPred);
end

