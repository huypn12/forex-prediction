# Forex price prediction
Machine Learning with Matlab 2018. Final project on Timeseries Prediction with LSTM / RNN.

## Caveats
As for prediction, this project is even worse than a naiive coin toss. While ARIMA/VAR model is stupidly simple yet showing some reasonable prediction, the LSTM model is an utter nonsense. The LSTM essentially does nothing but use the last observed tick as the 'prediction'; as the result, the predicted ticks, once charted, looks exactly like the test dataset, shifted one timestep to the future.

**To summarize, the so-called 'deep-learning' model in this project is a total heap of crap.**

Thus, this repository will be archived soon.

## Installation
In order to run the project, the following Matlab Toolboxes must be installed
1. Statistics and Machine Learning Toolbox
2. Econometrics Toolbox
3. Deep Learning Toolbox

## Running the Project
1. Add ml_proj and its subfolders into path
2. Modify the configuration if necessary. By default, it only verifies the saved models in full dataset. Set cfg.execMode to "train" to train the model again.
3. Run the respective .m file.


