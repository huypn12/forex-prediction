function MAPE = mape(YTest, YPred)
    MAPE = 100 * mae(YTest-YPred);
end

