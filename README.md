# Multiple Imputation Ensemble

[Multiple imputation ensembles](https://biostats.bepress.com/ucbbiostat/paper266/) (MIE) are a robust method for recovering missing data in classification problems. Here I apply MIE to the [Titanic competition](https://www.kaggle.com/c/titanic) on Kaggle. After variable selection and feature engineering, I used **multiple imputation with chain equations** ([MICE](https://pdfs.semanticscholar.org/dc64/aca1a942615fd932bc2b8e24f954b7a4d2c9.pdf)) to **train machine learning ensembles** of random forests and GBMs.

MICE is a flexible framework for imputing categorical and numerical data that estimates the predictive posterior distributions of missing data conditioned on existing data. Each iteration of MICE recovers missing data by random sampling from these distributions. The resulting imputations incorporate uncertainty estimates.

MICE has two considerations. First, data should be missing at random (MAR). I show how to use matrix plots to investigate. Second, MICE can be slow depending on the application. I show how to **run imputations in parallel** by initializing a **virtual cluster**.

When linking together many models trained on multiple imputations, the ensemble shows **greater diversity, better regularization, and reduced overfitting**. I tested two kinds of ensembles, random forests and gradient boosted machines.

In the first case I aggregated decision trees from 20 random forests trained on 20 imputations of the training data, then deployed this model to make predictions using 20 imputations of the testing data. Predictions were pooled and a majority vote was calculated. In the second case, I trained 400 GBMs, such that each imputation of the training data (20) was paired once with each imputation of the testing data (20). Again, I pooled predictions and computed the majority vote.

Random forests achieved 79% accuracy. I found that sub-sampling without replacement had better results than bootstrapping. Sub-sampling can optimize the bias-variance tradeoff.

Gradient boosted machines achieved 80% accuracy, which is in the top 7% of all submissions (1539/23618) on Kaggle.
