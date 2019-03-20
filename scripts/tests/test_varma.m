% Load data
eurusdCsvPath = 'EURUSD_15m_BID_sample.csv';
eurusd = readtable(eurusdCsvPath, 'Format', '%s%f%f%f%f%f');

eurusdOpen = eurusd.('Open');
eurusdOpen = eurusdOpen(:);
eurusdClose = eurusd.('Close');
eurusdClose = eurusdClose(:);
eurusdHigh = eurusd.('High');
eurusdHigh = eurusdHigh(:);
eurusdLow = eurusd.('Low');
eurusdLow = eurusdLow(:);

%% Divide in to train set and test set
ts = [eurusdOpen, eurusdHigh, eurusdLow, eurusdClose];
split = floor(size(ts,1)*0.8);
yTrain = ts(1:split, :);
yTest = ts(split+1:end, :);

%% Fit the model 
dimInput = size(yTrain, 2);
% minAic = Inf;
% minLag = 0;
% minResult = struct();
% for lag = 1:50
%     varmModel = varm(dimInput, lag);
%     estModel = estimate(varmModel, yTrain);
%     results = summarize(estModel);
%     aic = results.AIC;
%     if (aic < minAic)
%         minLag = lag;
%         minResult = results;
%     end
% end
lags = 23;
varmModel = var(dimInput, lag);
estModel = estimate(varmModel, yTrain);

%% Forecast
numPeriods = 5;
yForecast = zeros(size(yTest, 1), 4);

i = 1;
while i <= (size(yTest, 1) - lags)
    [tmp, ~] = forecast(estModel, numPeriods, yTest(i:(i + lags), :));
    for ii = i:(i+numPeriods-1)
       yForecast(ii, :) = tmp(ii - i + 1, :);
    end
    i = i + numPeriods;
end


