function RMSE = rmse(YTest, YPred)
    sdiff = (YTest - YPred).^2;
    RMSE = sqrt(sum(sdiff(:)/ numel(YTest)));
end
