# Model_Stacking_Titanic_Machine_Learning

This project showcases a machine learning solution to the Kaggle competition 'Titanic: Machine Learning from Disaster'. The goal of the competition was to predict who in the test data survived the sinking of the Titanic based on training data for 891 passengers. Data involved biographical details such as passenger class, age, and sex.

Variable selection, feature engineering, and multiple imputation were key to my success. I discarded variables with little benefit and extracted honorific titles from passenger names. I used multiple imputation with chained equations (MICE) to impute missing data.

To make predictions, I pooled votes from 20 ensembles trained on imputed data sets. I tested both random forests and gradient-boosted machines as candidate models. By pooling votes, I relied on a kind of model stacking.

Random forests achieved 79% accuracy. I found that sub-sampling without replacement yielded better results than bootstrapping. Sub-sampling can improve prediction accuracy by optimizing bias-variance tradeoff. 

Gradient boosted machines achieved 80% accuracy, which is in the top 8% of all submissions.
