%% LSTM Network to predict univariate time series
classdef lstm
    properties
        m_training_options;
        m_lstm_net;
    end
    
    methods
        function obj = lstm(...
                num_features, num_responses, num_hidden_units,...
                training_options)
            obj.m_lstm_net = [ ...
                sequenceInputLayer(num_features),...
                lstmLayer(num_hidden_units, 'OutputMode', 'sequence'),...
                lstmLayer(num_hidden_units, 'OutputMode', 'sequence'),...
                fullyConnectedLayer(num_responses),...
                regressionLayer
                ];
            obj.m_training_options = training_options;
        end
        
        function obj = train(obj)
            obj.m_lstm_net = trainNetwork(...
                x_train, y_train,...
                obj.m_lstm_net, training_options);
        end
       
        function model_name = name(obj)
            model_name = strcat(inputname(1), '_');
            model_name = strcat(model_name, datestr(now,'yyyymmddTHHMMSS'));
        end
       
    end
    
end
