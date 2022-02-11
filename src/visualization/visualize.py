import numpy as np


def show_random_forest_stats(result, expected):
    """ Showing stats related to the predicted and expected results"""
    error = abs(result - expected)  # Calculating the error
    avgAbsSrr = np.mean(error)      # Calculating average absolute error
    print('Average absolute error: {}%'.format(avgAbsSrr.values[0]*100))


def show_feature_importance(rf, features):
    """Showing the feature importance"""
    # Source for the below code segment: https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd
    
    feature_list = list(features.columns)
    
    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    
