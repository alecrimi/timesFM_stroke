![Static Plot](https://github.com/alecrimi/timesFM_stroke/blob/main/llmcausality.png)
![Animation Demo](https://github.com/alecrimi/timesFM_stroke/blob/main/bezierplot.gif)

This repository comprises the script used to run the Google Time Series Foundation Model to predict fMRI data and to create a causality framework Granger-like.

As in "Prediction and Causality of functional MRI and synthetic signal using a Zero-Shot Time-Series Foundation Model"
Alessandro Crimi, Andrea Brovelli https://arxiv.org/abs/2509.12497

The tested fMRI data are obtained from the data shared by Prof. CorbetTta:  10.1016/j.neuron.2015.02.027, data are not public due to privacy restrictions but can be obtained contacting the authors. 

The script main_stroke.ipynb compare the prediction in a zero-shot fashion from the foundation model to traditional time series predictors. 

finetuning_brain_control.ipynb and  finetuning_brain_patient.ipynb evaluate again the prediction but after some fine-tuning. 

MOUsimulations_with_multipletestingcorrection.ipynb is the script containing the causality analysis on netwoks with causal relationship according to Multivariate Ornstein-Uhlenbeck processes.

There is also a simpler test with logistic map time series at a chaotic regime
![Static Plot](https://github.com/alecrimi/timesFM_stroke/blob/main/logmap.png)
