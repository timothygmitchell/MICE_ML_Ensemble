# MICE_ML_Ensemble

This project showcases an ensemble machine learning solution to the Kaggle competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic). The goal was to predict who survived the sinking of the Titanic based on biographical details such as age, sex, and passenger class.

In addition to variable selection and feature engineering (discarding variables with little benefit and extracting honorific titles from passenger names), my solution involves **multiple imputation with chained equations** (MICE) to **diversify and regularize machine learning ensembles**.

MICE is a fast, flexible method for imputing numeric and categorical missing data. From a Bayesian perspective, missing values are assigned predictive posterior distributions conditioned upon existing data. Each iteration of MICE samples a random draw from these distributions. 

MICE [performs well in simulation studies](https://pdfs.semanticscholar.org/dc64/aca1a942615fd932bc2b8e24f954b7a4d2c9.pdf) but does come with a few concerns. First, it is sensitive to cases of data missing not at random (MNAR). I plot the data to flag any obvious violations. Second, it can be computationally slow for large data sets and chained random forests. However, it is easy to run imputations in parallel by starting a virtual cluster on a local machine.

To make predictions, I pooled votes from ensembles of tree-based models trained on imputed data sets. Since models were trained on multiple data sets, ensembles that integrate such models in a stochastic manner have better diversity, better regularizing properties, and are less sensitive to overfitting.

In the first case I aggregated decision trees from 20 random forests trained on 20 imputations of the training data, then made predictions on each of 20 imputations of the testing data. Predictions were combined using majority vote.

In the second case, I pooled votes from an ensemble of 400 GBM models, so that each imputation of the training data was paired once with each imputation of the testing data.

Random forests achieved 79% accuracy. I found that sub-sampling without replacement yielded better results than bootstrapping. Sub-sampling can optimize bias-variance tradeoff to enhance predictions.

Gradient boosted machines achieved 80% accuracy, which is in the top 7% of all submissions (1539/23618).
