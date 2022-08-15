# TreeAndLogisticRegressionModel

LOGIC:
First filters the data based on the categorical columns (choosed based on category_threshold) -> TREE PART
and then does logistic regression on the numerical columns after apply standard scaling -> REGRESSION PART
Idea is to first do tree part on the categorical columns and select the right rows from the train data based on 
these categorical values (test data) and then applying regression for numerical features.

#CAUTION:
Do LABEL ENCODING rather than ONE HOT ENCODING
Avoid PCA or SCALING
if PCA or scaling is done before -> then it is exactly same as logistic regression
