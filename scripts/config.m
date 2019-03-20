function cfg = config()
    %% Execution mode:
    %% "rerun" :
    %%     - ARIMA and VAR do grid search to confirm the optimal params.
    %%     - LSTM model is trained again.
    %% "verify" :
    %%     - ARIMA and VAR simply load the optimal params founded before.
    %%     - LSTM load the pretrained weights
    cfg.execMode = "rerun";
    %cfg.execMode = "verify";
    
    %% Dataset mode :
    %% "full"   : 245445 records, extremely computationally heavy
    %% "sample" : 14880 records, easier to compute
    cfg.dataset.mode = "full"; 
    %cfg.dataset.mode = "sample;
    cfg.dataset.trainSetRatio = 0.8; % ratio=0.8 -> train:val:test = 8:1:1
    
    %% ARIMA saved optimal parameters
    cfg.arima.params = {...
        20 0 6;... %% Params for Open
        20 0 6;... %% Params for High
        20 0 6;... %% Params for Low
        20 0 6;... %% Params for Close
        };
    
    %% VAR saved optimal parameters
    cfg.var.P = 53;
    
    %% LSTM uni variate saved weight
    cfg.lstm.uni.openModelFile = "";
 
    %% LSTM multivariate saved weight
    cfg.lstm.multi.openModelFile = "";
    
    %% Rerun parameter, max lags
    cfg.lstm.maxlags = 50;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    csvPathFull = "EURUSD_15m_BID_01.01.2010-31.12.2016.csv";
    csvPathSample = "EURUSD_15m_BID_sample.csv";
    if cfg.dataset.mode == "full"
        cfg.dataset.csvPath = csvPathFull;
    else
        cfg.dataset.csvPath = csvPathSample;
    end
    
end
