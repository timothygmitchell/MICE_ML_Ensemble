# MICE_ML_Ensemble

This project showcases an ensemble machine learning solution to the Kaggle competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic). My solution involves a unique, robust approach to recovering missing data that [has only recently been documented in the machine learning literature](https://biostats.bepress.com/ucbbiostat/paper266/).

Along with variable selection and feature engineering, I used **multiple imputation with chained equations** ([MICE](https://pdfs.semanticscholar.org/dc64/aca1a942615fd932bc2b8e24f954b7a4d2c9.pdf)) to **train machine learning ensembles**.

MICE is a flexible framework for imputing categorical and numeric data. In Bayesian terms, MICE assigns predictive posterior distributions to missing value conditioned upon existing data. Each iteration of MICE results in a random draw from theoretical distributions. The resulting imputations incorporate uncertainty estimates.

MICE has two considerations. First, data should be missing at random (MAR). I show how to use matrix plots to investigate. Second, MICE is much slower for big data sets. I show how to **run imputations in parallel by initializing a virtual cluster**.

Since models were trained on multiple imputations, ensembles of such models show **greater diversity, better regularizing properties, and less sensitivity to overfitting**. I tested two kinds of ensembles, random forests and gradient boosted machines, and pooled together votes to make predictions.

In the first case I aggregated decision trees from 20 random forests trained on 20 imputations of the training data, then made predictions using 20 imputations of the testing data. In the second case, I trained 400 GBMs, so that each imputation of the training data (20) was paired once with each imputation of the testing data (20).

Random forests achieved 79% accuracy. I found that sub-sampling without replacement had better results than bootstrapping. Sub-sampling can optimize the bias-variance tradeoff.

Gradient boosted machines achieved 80% accuracy, which is in the top 7% of all submissions (1539/23618).
