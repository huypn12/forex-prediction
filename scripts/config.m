function cfg = config()
    %% Execution mode:
    %% "train" :
    %%     - ARIMA and VAR do grid search to confirm the optimal params.
    %%     - LSTM model is trained again.
    %cfg.execMode = "train";
    %% "verify" :
    %%     - ARIMA and VAR simply load the optimal params founded before.
    %%     - LSTM load the pretrained weights
    cfg.execMode = "verify";
    
    %% Dataset mode :
    %% "full"   : 245445 records, extremely computationally heavy
    cfg.dataset.mode = "full"; 
    %% "sample" : 14880 records, easier to compute
    %cfg.dataset.mode = "sample";
    %% Split ratio, e.g. ratio=0.8 -> train:val:test = 8:1:1
    cfg.dataset.trainSetRatio = 0.8; 
    
    %% Maximum lags (how many timesteps we look back). 
    %% Used by VARM and ARIMA during parameter search (training).
    cfg.numLags = 25; %% Higher than 25 leads to LONG time VAR/ARIMA params search
        
    %% ARIMA saved models
    cfg.arima.savedModelsFile = "arima_models";
    
    %% VAR saved models
    cfg.varm.savedModelsFile = "varm_models";
    
    %% LSTM uni variate saved weight
    cfg.lstm.uni.savedModelsFile = "lstm_univariate_models";
 
    %% LSTM multivariate saved weight
    cfg.lstm.multi.savedModelsFile = "lstm_multivariate_models";
        
    %% Finish producing configuration
    cfg.numResponses = 1; %% Multiple timesteps ahead is not tested
    csvPathFull = "EURUSD_15m_BID_01.01.2010-31.12.2016.csv";
    csvPathSample = "EURUSD_15m_BID_sample.csv";
    if cfg.dataset.mode == "full"
        cfg.dataset.csvPath = csvPathFull;
    else
        cfg.dataset.csvPath = csvPathSample;
    end
    
end
