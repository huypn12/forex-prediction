function [trainset, valset, testset] = eurusdPartition(eurusd, ratio)
    
    valSplitIdx = floor(ratio * size(eurusd, 1));
    testSplitIdx = valSplitIdx + floor(0.5*(1-ratio)*size(eurusd, 1));
    
    trainset = eurusd(1:valSplitIdx, :);
    valset = eurusd((valSplitIdx+1):testSplitIdx, :);
    testset = eurusd((testSplitIdx+1):end, :);
    
end
