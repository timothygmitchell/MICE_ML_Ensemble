# MICE_ML_Ensemble

This project showcases an ensemble machine learning solution to the Kaggle competition 'Titanic: Machine Learning from Disaster'. The goal of the competition was to predict who in the test data survived the sinking of the Titanic based on training data for 891 passengers. Data included biographical details such as age, sex, and passenger class.

Data pre-processing involved variable selection, feature engineering, and multiple imputation. I discarded variables with little benefit and extracted honorific titles from passenger names. I used multiple imputation with chained equations (MICE) to impute missing data.

To make predictions, I pooled votes from 20 ensembles trained on imputed data sets. I tested both random forests and gradient-boosted machines as candidate models. By pooling votes, I relied on a kind of model stacking.

Random forests achieved 79% accuracy. I found that sub-sampling without replacement yielded better results than bootstrapping. Sub-sampling can improve prediction accuracy by optimizing bias-variance tradeoff. 

Gradient boosted machines achieved 80% accuracy, which is in the top 7% of all submissions (1539/23618).
