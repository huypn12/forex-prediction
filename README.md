# ml_proj

Machine Learning with Matlab 2018. Final project on Timeseries Prediction with LSTM / RNN.

0. Project directory structure:

ml_proj/
+--	data		% Contains EURUSD dataset
+--	scripts		% MATLAB Scripts
|	+-- data		   % data preprocessing scripts
|	+-- measure		   % RMSE and MAPE implementation
|	+-- arimaMain.m		   % Run ARIMA
|	+-- varmMain.m		   % Run VAR
|	+-- lstmUnivariateMain.m   % Run LSTM univariate
|	+-- lstmMultivariateMain.m % Run LSTM multivariate
|	+-- config.m 		   % Configuration
+--	reports		% Report (report.pdf)
+--	saved_models 	% Trained models

1. Installation
	In order to run the project, the following Matlab Toolboxes must be installed: 
		1. Statistics and Machine Learning Toolbox
		2. Econometrics Toolbox
		3. Deep Learning Toolbox

2. Running the Project
	- Add ml_proj and its subfolders into path
	- Modify the configuration if necessary. By default, it only verifies the saved models in full dataset. Set cfg.execMode to "train" to train the model again.
	- Run the respective .m file
