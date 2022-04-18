
# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from pandas.tseries.offsets import DateOffset
from sklearn.inspection import permutation_importance
# from sklearn.metrics import classification_report


def get_train_split(X, y):
    """
    Get the train split for the machine learning model and scales the data.
    Returns scaled and non-scaled trained and test data.
    """
    # Select the start of the training period
    training_begin = X.index.min()

    # Display the training begin date
    print(training_begin)
    # Select the ending period for the training data with an offset of 3 months
    training_end = X.index.min() + DateOffset(months=6)

    # Display the training end date
    print(training_end)

    # Generate the X_train and y_train DataFrames
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]

    # Display sample data
    X_train.head()

    # Generate the X_test and y_test DataFrames
    X_test = X.loc[training_end:]
    y_test = y.loc[training_end:]

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
    # Scaled data
    rdm_forest_model = RandomForestClassifier(max_depth=10, random_state=None)
    rdm_forest_model.fit(X_train_scaled, np.ravel(y_train, order='c'), sample_weight=None)
    # rdm_forest_model.fit(X_train_scaled, y_train.values.ravel(), sample_weight=None)
    feat_importance = rdm_forest_model.feature_importances_
    count = 0
    X_new = X.copy()
    X_new_cols = X_new.columns.to_list()
    if 'SPY' in X_new_cols:
        X_new.drop(columns={'SPY'})

    print('X_new_cols', X_new_cols)
    columns_to_drop = []
    dropped_feature_importances = []
    # Check for importance level and remove cols from df below threshold
    for each_feat in feat_importance:
        if each_feat < 0.085:
            dropped_feature_importances.append(each_feat)
            columns_to_drop.append(X_new_cols[count])
            # Remove open and close columns from X_new
            X_new.drop(columns={X_new_cols[count]}, inplace=True)
        count = count + 1

    # check new X df for accuracy
    print('feat_importance\n', feat_importance)
    print('dropped_feature_importances\n', dropped_feature_importances)
    print('dropped_X_columns\n', columns_to_drop)
    
    # Return the model and the new X df with optimized important columns
    return {
        'rdm_forest_model': rdm_forest_model,
        'X_new': X_new
    }