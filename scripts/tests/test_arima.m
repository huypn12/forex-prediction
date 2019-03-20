% Load data
eurusdCsvPath = 'EURUSD_15m_BID_sample.csv';
eurusd = readtable(eurusdCsvPath, 'Format', '%s%f%f%f%f%f');
euOpen = eurusd.('Open');
euOpen = euOpen(:);
euClose = eurusd.('Close');
euClose = euClose(:);
euHigh = eurusd.('High');
euHigh = euHigh(:);
euLow = eurusd.('Low');
euLow = euLow(:);


% estimate 
% minAic = Inf;
% minParams = array();
% for i = 1:10
%     for j = 1:10
%         for k = 1:10
%             mdl = arima(i, j, k);
%             estModel = estimate(mdl, yTrain);
%             results = summarize(estModel);
%             aic = results.AIC;
%             if (aic < minAic)
%                 minAic = aic;
%                 minParams = [i j k];
%                 minResult = results;
%             end
%         end
%     end
% end

split = floor(size(euOpen, 1) * 0.8);
yTrain = euOpen(1:split);
yTest = euOpen(split+1:end);

mdl = arima(5, 0, 6);
estModel = estimate(mdl, yTrain);

lags = 6;
numPeriods = 1;
yForecast = zeros(size(yTest, 1), 1);

i = 1;
while i <= (size(yTest, 1) - lags)
    [tmp, ~] = forecast(estModel, numPeriods, 'Y0', yTest(i:(i + lags)));
    yForecast(i + lags) = tmp; 
    i = i + 1;
end


%% Calculate the error
rmseErr = rmse(yForecast, yTest);
mapeErr = mape(yForecast, yTest);

%% Plot
figure
plot(yTest(1:2000))
hold on
plot(yForecast(1002:2000),'.-')
hold off
legend(["Observed" "Forcast"])
ylabel("EURUSD Open (BUY)")
title("ARIMA(5,0,6), RMSE = " + rmseErr + "; MAPE = " + mapeErr)

