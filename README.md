# AutoML

Selecting model architectures and hyperparameters is often a difficult task in applied machine learning. Automated machine learning (AutoML) is a framework of tools designed to automate this model selection process. Performance predictors have the potential to be useful in this process [1,2,3]. Performance predictors take a dataset and hyperparameters as input and predict the performance of
the resulting trained model. Thus, an effective performance predictor can be used to select good model architectures and hyperparameters, e.g., since the learned predictor is a function, one could create approaches for optimizing it. In this project, the goal will be to train a performance predictor. Each example in the training set is a network architecture, associated meta-data (details below), and the train and test performance of the associated model when it was trained to convergence. 

This project develops a supervised machine learning solution that takes a model architecture and hyperparameters and predicts the training and testing performance. The approach is to first preprocess the data (make categorical numeric, normalize, remove unneeded features etc). The file train.csv is split using a train/validation split to train four regressors: XGBoost, a Random Forest, and two neural networks. Afterwards the regressors are refit using all of train.csv as training data. Final predictions is an ensemble of each algorithm's predictions on test.csv.

[1] Neural Architecture Optimization, https://arxiv.org/abs/1808.07233 \\
[2] Progressive Neural Architecture Search, https://arxiv.org/abs/1712.00559 \\
[3] Accelerating Neural Architecture Search using Performance Prediction, https://arxiv.org/abs/1705.10823
[4] A Surprising Linear Relationship Predicts Test Performance in Deep Networks, https://arxiv.org/abs/1807.09659
[5] Predicting the Generalization Gap in Deep Networks with Margin Distributions, https://arxiv.org/abs/1810.00113
