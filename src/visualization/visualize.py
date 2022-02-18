import numpy as np
from sklearn.metrics import roc_auc_score


def show_random_forest_stats(rf, test_X, y_test, rf_probs):
    """ Showing stats related to the predicted and expected results
    rf: Trained random forest model
    test_X: training set
    y_test: Actual labels 
    rf_probs: Probability of selected label (in this case 1)
    """

    # Showing the mean error
    print('Average absolute error: {}%'.format((1-rf.score(test_X, y_test))*100))

    # Showing the ROC
    print('ROC: {:0.5f}'.format(roc_auc_score(y_test, rf_probs)))


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
    
