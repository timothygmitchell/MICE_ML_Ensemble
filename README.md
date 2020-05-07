# TitanicMachineLearning

This project showcases a machine learning solution to the Titanic Kaggle competition. I used feature engineering to derive variables and multiple imputation with chained equations (MICE) to impute missing data.

For predictions, I pooled votes from tree-based ensembles grown using imputed data sets. 

Random forests achieved an accuracy of about 79%, and I found that sub-sampling without replacement yielded better results than bootstrapping. This may be an optimization of the bias-variance tradeoff. 

Gradient boosted machines achieved an accuracy of 80%, which places me in the top 8% of submissions. I found that a sample fraction of 0.9 yielded the best results.
