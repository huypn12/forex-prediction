function cfg = config()
    %% Execution mode:
    %% "rerun" :
    %%     - ARIMA and VAR do grid search to confirm the optimal params.
    %%     - LSTM model is trained again.
    %% "verify" :
    %%     - ARIMA and VAR simply load the optimal params founded before.
    %%     - LSTM load the pretrained weights
    cfg.execMode = "train";
    %cfg.execMode = "verify";
    
    %% Dataset mode :
    %% "full"   : 245445 records, extremely computationally heavy
    %% "sample" : 14880 records, easier to compute
    %cfg.dataset.mode = "full"; 
    cfg.dataset.mode = "sample";
    cfg.dataset.trainSetRatio = 0.8; % ratio=0.8 -> train:val:test = 8:1:1
    
    %% Maximum lags (how many timesteps we look back), used by all models.
    cfg.maxLags = 3;
    
    %% ARIMA saved optimal parameters
    cfg.arima.params = {...
        20 0 6;... %% Params for Open
        20 0 6;... %% Params for High
        20 0 6;... %% Params for Low
        20 0 6;... %% Params for Close
        };
    cfg.arima.savedModels = "";
    
    %% VAR saved optimal parameters
    cfg.var.P = 53;
    cfg.varm.savedModel = "";
    
    %% LSTM uni variate saved weight
    cfg.lstm.uni.savedModels = "";
 
    %% LSTM multivariate saved weight
    cfg.lstm.multi.openModelFile = "";
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    csvPathFull = "EURUSD_15m_BID_01.01.2010-31.12.2016.csv";
    csvPathSample = "EURUSD_15m_BID_sample.csv";
    if cfg.dataset.mode == "full"
        cfg.dataset.csvPath = csvPathFull;
    else
        cfg.dataset.csvPath = csvPathSample;
    end
    
end
