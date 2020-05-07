# TitanicMachineLearning

This project showcases a machine learning solution to the Kaggle competition 'Titanic: Machhine Learning from Disaster'. I used feature engineering to derive variables and multiple imputation with chained equations (MICE) to impute missing data.

I chose random forests and gradient-boosted machines as candidate models. To make predictions, I pooled votes from individual ensembles trained using imputed data sets. 

Random forests achieved an accuracy of 79%. I found that sub-sampling without replacement yielded better results than bootstrapping. This was possibly an optimization of the bias-variance tradeoff. 

Gradient boosted machines achieved an accuracy of 80%, which is in the top 8% of all submissions.
