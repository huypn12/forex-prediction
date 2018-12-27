%% Load raw FX data

classdef fx_data
    properties
        m_timestamp;
        m_open;
        m_high;
        m_low;
        m_close;
        m_volume;
        m_is_standardized;
    end
    
    
    methods
        function obj = fx_data(csv_path, timestamp_format, to_standardize)
            fx_data = load_fx_data(csv_path, timestamp_format, to_standardize);
            obj.m_open = fx_data.('open');
            obj.m_close = fx_data.('close');
            obj.m_high = fx_data.('high');
            obj.m_low = fx_data.('low');
            
            obj.m_timestamp = fx_data.('time');
            obj.m_volume = fx_data.('volume');
        end
        
    end
    
    methods(Static)
        function fx_data = load_fx_data(...
                csv_path, timestamp_format, to_standardize)
            
            fx_data = readtable(csv_path, 'Format', '%s%f%f%f%f%f');
            
            if ~isempty(timestamp_format)
                timestamps = fx_data.('time');
                timestamps = arrayfun(...
                    @(dt) datetime(dt, 'InputFormat', timestamp_format),...
                    timestamps);
                fx_data.('time') = timestamps;
            end
            
            if to_standardize
                for field_name = ['open', 'high', 'low', 'close']
                    fx_data.(field_name) = ts_standardize(fx_data.(field_name));
                end
            end
        end
        
        function ts = ts_standardize(ts_array)
            mu = mean(ts_array);
            sigma = std(ts_array);
            ts = (ts_array - mu) / sigma;
        end
        
        %% @param ratio: e.g. [8 1 1] array of 3 integers sum up to 10
        %% @return array of 3 fx_data object
        function [train_data, valid_data, test_data] = ...
                split(fx_data_obj, ratio)
            
            
        end
    end
end

