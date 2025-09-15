This repository comprises the script used to run the Google Time Series Foundation Model to predict fMRI data and to create a causality framework Granger-like.

The tested fMRI data are obtained from the data shared by Prof. Corbetta:  10.1016/j.neuron.2015.02.027, data are not public due to privacy restrictions but can be obtained contacting
the authors. 

The script main_stroke.ipynb compare the prediction in a zero-shot fashion from the foundation model to traditional time series predictors. 

finetuning_brain_control.ipynb and  finetuning_brain_patient.ipynb evaluate again the prediction but after some fine-tuning. 

MOUsimulations_with_multipletestingcorrection.ipynb is the script containing the causality analysis on netwoks with causal relationship according to Multivariate Ornstein-Uhlenbeck processes.
