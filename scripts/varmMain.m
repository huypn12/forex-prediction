%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cfg = config();

[eurusd, featureNames] = eurusdDataset(cfg.dataset.csvPath, "");
[eurusdTrain, ~, eurusdTest] = eurusdPartition(...
    eurusd, cfg.dataset.trainSetRatio);

if cfg.execMode == "rerun"
    [estMdl, YPred, aic, rmse, mape] = appVarmRerun(YTrain, YTest);
else
    p = cfg.var.P;
    [estMdl, YPred, aic, rmse, mape] = appVarmVerify(YTrain, YTest, P);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [varMdl, YPred, aic, rmse, mape] = appVarRerun(YTrain, YTest)
    [varMdl, aic] = varmSearchParams(YTrain);
    [YPred, rmse, mape] = varmPredict(YTest, 1);
    visualizeResult(varMdl, YPred, YTest, rmse, mape);
    
end

function [YPred, aic, rmse, mape] = appVerify(YTrain, YTest, P)
    mdl = varm(dimInput, P);
    estMdl = estimate(mdl, YTrain);
    results = summarize(estMdl);
    aic = results.AIC;
    
    
end


function [varMdl, aic] = varmSearchParams(YTrain)
    dimInput = size(YTrain, 2);
    
    minAic = Inf;
    minMdl = varm(dimInput, 1);
    
    for lag = 1:60
        mdl = varm(dimInput, lag);
        estMdl = estimate(mdl, YTrain);
        results = summarize(estMdl);
        aic = results.AIC;
        if (aic < minAic)
            minAic = aic;
            minMdl = estMdl;
        end
    end
    
    varMdl = minMdl;
end

function visualizeResult(estMdl, YTest, YPred, mape, rmse)
    %% Plot the predicted value, observed value, and error
    modelDesc = strcat("VAR(", num2str(estMdl.P), ")");
    figureTag = strcat(modelDesc + " on EURUSD BID, ", featureName, " price");
    figure('Name', figureTag);
    candle(YTest(:, 500:1000).', 'b')
    hold on
    candle(YPred(:, 500:1000).', 'r')
    hold off
    ylabel("EURUSD(BUY)")
    title("RMSE=" + rmse + " MAPE=" + mape);
end

function [YPred, mape, rmse] = varmPredict(estMdl, Ytest, numResponse)
end