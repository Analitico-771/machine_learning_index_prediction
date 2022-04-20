
# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from pandas.tseries.offsets import DateOffset
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report


def get_train_split(X, y):
    """
    Get the train split for the machine learning model and scales the data.
    Returns scaled and non-scaled trained and test data.
    """
    # Select the split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Display sample data
    X_test.head()

    # Create a StandardScaler instance
    scaler = StandardScaler()
    
    # Apply the scaler model to fit the X-train data
    X_scaler = scaler.fit(X_train)
    
    # Transform the X_train and X_test DataFrames using the X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    return {
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test
    }


def get_importance(train_split, X):
    """
    Get the importance of the df features and returns a new df
    with only the selected important columns as features.
    Returns the RandomForestClassifier model instance.
    """
    X_test, X_train_scaled, X_test_scaled, y_train, y_test = train_split.values()
    # Create an instance of the model
    rdm_forest_model = RandomForestClassifier(max_depth=5, random_state=None)
    rdm_forest_model.fit(X_train_scaled, np.ravel(y_train, order='c'), sample_weight=None)
    # analyze the feature importance values
    feat_importances = rdm_forest_model.feature_importances_
    new_feature_importances = []
    columns_to_drop = []
    dropped_feature_importances = []
    count = 0
    X_new = X.copy()
    X_new_cols = X_new.columns.to_list()
    if 'SPY' in X_new_cols:
        X_new.drop(columns={'SPY'})

    print('X_new_cols', X_new_cols)
    
    importance = 0.095
    # print(np.mean(feat_importances))
    # Check for importance level and remove cols from df below threshold
    for each_feat in feat_importances:
        if each_feat <= importance:
            dropped_feature_importances.append(each_feat)
            # new_feature_importances.pop(each_feat)
            columns_to_drop.append(X_new_cols[count])
            # Remove open and close columns from X_new
            X_new.drop(columns={X_new_cols[count]}, inplace=True)
        elif each_feat > importance:
            new_feature_importances.append(each_feat)
        count = count + 1

    # check new X df for accuracy
    print('feat_importances\n', feat_importances)
    print('new_feature_importances\n', new_feature_importances)
    print('dropped_feature_importances\n', dropped_feature_importances)
    print('dropped_X_columns\n', columns_to_drop)
    
    # Return the model and the new X df with optimized important columns
    return {
        'new_feature_importances': new_feature_importances,
        'rdm_forest_model': rdm_forest_model,
        'X_new': X_new
    }