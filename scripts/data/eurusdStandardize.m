%% Preprocessing, dataset passed into this part must be row-prioritized
%% N-k, where N is the number of records and k is number of dimension

function stdX= eurusdStandardize(eurusd)
    stdX = zeros(size(eurusd));
    for i = 1:size(eurusd, 2)
        stdX(:, i) = standardize(eurusd(:, i));
    end
end

function X = standardize(X_)
    X = (X_ - mean(X_))/std(X_);
end


