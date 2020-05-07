# TitanicMachineLearning

This project showcases a machine learning solution to the Kaggle competition 'Titanic: Machine Learning from Disaster'. The goal of the competition was to predict who in the test data survived based on training data containing biographical details and survival outcomes for 891 passengers.

Variable selection, feature engineering, and multiple imputation were key stages of my analysis. I discarded variables with little benefit and extracted honorific titles from passenger names. I used multiple imputation with chained equations (MICE) to impute missing data.

To make predictions, I pooled votes from 20 ensembles trained on imputed data sets. I tested random forests and gradient-boosted machines as candidate models. 

Random forests achieved 79% accuracy. I found that sub-sampling without replacement yielded better results than bootstrapping. Sub-sampling can improve prediction accuracy by optimizing bias-variance tradeoff. 

Gradient boosted machines achieved 80% accuracy, which is in the top 8% of all submissions.
