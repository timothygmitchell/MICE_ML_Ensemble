# MICE_ML_Ensemble

This project showcases an ensemble machine learning solution to the Kaggle competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic). My solution is unique in several ways.

Besides variable selection and feature engineering, my solution involved **multiple imputation with chained equations** (MICE) to **diversify and regularize machine learning ensembles**.

MICE is a flexible framework for imputing categorical and numeric data. From a Bayesian perspective, MICE assigns predictive posterior distributions to missing data conditioned upon existing data. Each iteration of MICE results in a random draw from theoretical distributions. The resulting imputations incorporate probability estimates.

MICE [performs well in simulation studies](https://pdfs.semanticscholar.org/dc64/aca1a942615fd932bc2b8e24f954b7a4d2c9.pdf) but it does have certain considerations. First, data should be missing at random (MAR). I show how to use matrix plots to investigate. Second, MICE does not always scale for large data sets. I show how to **run imputations in parallel by initializing a virtual cluster**.

For predictions, I pooled votes from ensembles of tree-based models. Since models were trained on multiple imputations, the resulting ensembles had greater diversity and better regularizing properties.

In the first case I aggregated decision trees from 20 random forests trained on 20 imputations of the training data, then made predictions using 20 imputations of the testing data. In the second case, I trained 400 GBM models, so that each imputation of the training data was paired once with each imputation of the testing data during the train/test procedure.

Random forests achieved 79% accuracy. I found that sub-sampling without replacement had better results than bootstrapping. Sub-sampling can optimize the bias-variance tradeoff.

Gradient boosted machines achieved 80% accuracy, which is in the top 7% of all submissions (1539/23618).
