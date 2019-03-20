function [eurusd, features] = eurusdDataset(csvPath, csvFormat)
    
    % Load csv file
    if csvFormat == ""
        csvFormat = "%s%f%f%f%f%f";
    end
    forexRates = readtable(csvPath, "Format", csvFormat);
    
    openPrice = forexRates.("Open");
    openPrice = openPrice(:);
    closePrice = forexRates.("Close");
    closePrice = closePrice(:);
    highPrice = forexRates.("High");
    highPrice = highPrice(:);
    lowPrice = forexRates.("Low");
    lowPrice = lowPrice(:);
    volume = forexRates.("Volume");
    volume = volume(:);
    
    %% Median, Mean, Momentum features were added to later train LSTM model better
    medianPrice = 0.5 * (highPrice + lowPrice);
    meanPrice = 0.25 * (highPrice + lowPrice + openPrice + closePrice);
    momentum = volume .* (openPrice - closePrice);
    
    eurusd = [openPrice, highPrice, lowPrice, closePrice, volume,...
        medianPrice, meanPrice, momentum];
    features = ["open", "high", "low", "close", "volume",...
        "median", "mean", "momentum"];
    
end