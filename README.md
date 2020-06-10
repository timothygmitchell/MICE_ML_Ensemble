# MICE_ML_Ensemble

This project showcases an ensemble machine learning solution to the Kaggle competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic). The goal was to predict who survived the sinking of the Titanic based on biographical details such as age, sex, and passenger class.

Besides variable selection and feature engineering (discarding variables with little benefit, extracting honorific titles from passenger names), my solution involved **multiple imputation with chained equations** (MICE) to **diversify and regularize machine learning ensembles**.

MICE is a fast, flexible method for imputing numeric and categorical missing data. From a Bayesian perspective, missing values are assigned predictive posterior distributions conditioned upon existing data. Each iteration of MICE samples from these distributions. 

MICE [performs well in simulation studies](https://pdfs.semanticscholar.org/dc64/aca1a942615fd932bc2b8e24f954b7a4d2c9.pdf) but does involve certain considerations. First, data need to be missing at random (MAR) for optimal performance. I show that it is easy to flag obvious violations with matrix plots. Second, MICE can be computationally slow. I show it is easy to run imputations in parallel by initializing a virtual cluster.

To make predictions, I pooled votes from ensembles of tree-based models. Since models were trained on multiple imputations, the resulting ensembles showed better diversity, better regularizing properties, and less sensitivity to overfitting.

In the first case I aggregated decision trees from 20 random forests trained on 20 imputations of the training data, then made predictions for each of 20 imputations of the testing data. Predictions were averaged by majority vote.

In the second case, I pooled votes from an ensemble of 400 GBM models, so that each imputation of the training data was paired once with each imputation of the testing data during the train/test process.

Random forests achieved 79% accuracy. I found that sub-sampling without replacement had better results than bootstrapping. Sub-sampling can optimize bias-variance tradeoff to reduce generalization error.

Gradient boosted machines achieved 80% accuracy, which is in the top 7% of all submissions (1539/23618).
