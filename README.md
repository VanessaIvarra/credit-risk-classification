# credit-risk-classification

## Overview of the Analysis

Below is a brief overview of the analysis performed for credit-risk-classification:

* The purpose of this activity is to train and evaluate machine learning models based on loan risk. The activity uses a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers and classify the credit risk predictions.

* The target financial information in the data is the loan status. The features in the financial data that are used to predict the loan status are loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks and total debt. 

* For the target values of `loan_status` there are two variables which I have tried to predict. The first set of variables have a value of '0' (value count of 75036) and these indicate that the loan is healthy. The second set of variables have a value of '1' (value count of 2500) and these indicate that the loan has a high risk of defaulting.

* The machine learning process used to perform the analysis has following stages:
 - Create label sets and feature Dataframe from the provided dataset.
 - Split the data into training and testing datasets by using train_test_split.
 - Create a logistic regression model and fit our original data into the model.
 - Make predictions on testing data labels by using the testing feature data and the fitted model.
 - Evaluate the model's performance by generating a confusion matrix and classification report.

* First method used in this case is the `LogisticRegression` model on the original fitted data. As the data was highly overweighted towards one of the target variables (healthy loans), so `RandomOverSampler` was used to reduce the imbalances and LogisticRegression was then applied to the oversampled data.