# TitanicMachineLearning

This project showcases a machine learning solution to the Kaggle competition 'Titanic: Machine Learning from Disaster'. The data set contains biographical details for several hundred passengers onboard the Titanic. The goal of the competition was to predict which passengers in the testing data set survived.

Variable selection and feature engineering were key steps in my analysis. I used multiple imputation with chained equations (MICE) to impute missing data.

I tested random forests and gradient-boosted machines as candidate models. To make predictions, I pooled votes from 20 ensembles trained on imputed data sets. 

Random forests achieved 79% accuracy. I found that sub-sampling without replacement yielded better results than bootstrapping. This was likely an optimization of the bias-variance tradeoff. 

Gradient boosted machines achieved 80% accuracy, which is in the top 8% of all submissions.
