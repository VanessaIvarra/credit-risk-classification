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

## Results

This section describes the results of the machine learning model used for the activity:

* Machine Learning Model 1 - **Logistic Regression**:

Below are the key results that can be drawn from the results computed by the Logistic Regression model that was performed on the original fitted data:

  * This model does a good job in predicting both the healthy and the high-risk loans.

  * This model has a precision score of 100% for the healthy loans and 84% for the high-risk loans. This again can be attributed to the imbalance in the data. The precision scores imply that the healthy loans were classified correctly as positive 100% of the times. However, for the high-risk loans, the classification was correct only 85% of the times.

  * This model has a recall score of 99% for the healthy loans and 91% for the high-risk loans. The scores imply that for all the instances where the loans were actually healthy, 99% of the times they were classified correctly. However, for all the instances where the loans were actually high-risk, they were classified correctly 91% of the times.

## Summary

Based on the results described above, I find that:

* It looks like that for any lending service company or credit grantor, the correct identification of a high-risk loan is much more important as it reduces the probabilty of defaultors by placing importance on correct analysis of their creditworthiness, thereby impacting the overall profitability of the company. The oversampled model does a way better job in correct classification of the high-risk loans(99%) when they were actually positive by catching the incorrect labelling of high-risk loans as healthy.

My recommendation would be to see what methods can be use with this data set first before resulting to this one. The reason being for this is because if there is an imbalance in the data then that could really affect the outcome of your data. 