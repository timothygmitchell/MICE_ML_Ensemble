# TitanicMachineLearning

This project showcases a machine learning solution to the Kaggle competition 'Titanic: Machine Learning from Disaster'. The goal of the competition was to predict who in the test data survived based on training data containing biographical details and survival status for several hundred passengers.

Variable selection and feature engineering were key steps in my analysis. I used multiple imputation with chained equations (MICE) to impute missing data.

I tested random forests and gradient-boosted machines as candidate models. To make predictions, I pooled votes from 20 ensembles trained on imputed data sets. 

Random forests achieved 79% accuracy. I found that sub-sampling without replacement yielded better results than bootstrapping. Sub-sampling can optimize bias-variance tradeoff. 

Gradient boosted machines achieved 80% accuracy, which is in the top 8% of all submissions.
