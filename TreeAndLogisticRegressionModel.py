class TreeAndLogisticRegressionModel():
    """
    Do LABEL ENCODING rather than ONE HOT ENCODING
    Avoid PCA or SCALING
    if PCA or scaling is done before -> then it is exactly same as logistic regression
    
    LOGIC:
    First filters the data based on the categorical columns (choosed based on category_threshold) -> TREE PART
    and then does logistic regression on the numerical columns after apply standard scaling -> REGRESSION PART
    Idea is to first do tree part on the categorical columns and select the right rows from the train data based on 
    these categorical values (test data) and then applying regression for numerical features.
    """
    
    def __init__(self, category_threshold=5):
        self.category_threshold = category_threshold
        self.fit_df = None
        self.categorical_columns = None
        self.numerical_columns = None
    
    def fit(self, X, y):
        category_threshold = self.category_threshold
        categorical_columns = [col for col in X.columns if len(X[col].unique())<=category_threshold]
        numerical_columns = [col for col in X.columns if col not in categorical_columns]
        X['y'] = np.array(y)
        self.fit_df = X
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        return self
        
    def predict_proba(self, X):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        category_threshold = self.category_threshold
        categorical_columns = self.categorical_columns
        numerical_columns = self.numerical_columns
        y_prob = []
        y_count = []
        X.reset_index(drop=True, inplace=True)
        for i, row in X.iterrows():
            fit_df = self.fit_df
            row_df = X.iloc[[i]]
            row_df = row_df.loc[:,[col for col in row_df.columns if col in numerical_columns]]
            #this step is for TREE part of the code -> for categorical columns
            for col in X.columns:
                #same as (col in numerical columns)
                if col not in categorical_columns:
                    continue
                else:
                    fit_df = fit_df.loc[fit_df[col]==row[col]]
                    fit_df.drop(columns = col, inplace= True)
            #this step is for REGRESSION part of the code -> for numerical columns
            X_fit = fit_df.loc[:, fit_df.columns!='y']
            y_fit = fit_df['y']
            #scaling the data before using linear regression
            scaler = StandardScaler()
            scaler.fit(X_fit)
            X_fit = pd.DataFrame(scaler.transform(X_fit))
            row_df = pd.DataFrame(scaler.transform(row_df), columns = X_fit.columns)
            logistic_regression_model = LogisticRegression()
            logistic_regression_model.fit(X_fit,y_fit)
            try:
                prob = (logistic_regression_model.predict_proba(row_df)[0])
            except:
                prob = np.nan
            y_prob.append(prob)
            y_count.append(len(X_fit))
        return y_prob #, y_count
            
    def get_params(self):
        return self.fit_df, self.categorical_columns, self.numerical_columns
            
        