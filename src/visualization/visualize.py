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
    
    
    
def plot_confusion_matrix_full(model,X_train, y_train, X_val, y_val):
    """Score Models and return results as a dataframe

    Parameters
    ----------
    model: model
        Model passed into function
    X_train : Numpy Array
        X_train data
    y_train : Numpy Array
        Train target
    X_val : Numpy Array
         X_val data
    y_val : Numpy Array
        Val target        
    
    Returns
    -------
    """
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    
    
    
    # Plot non-normalized confusion matrix
    titles_options = [
        ("Train - Confusion matrix, without normalization", None, X_train, y_train),
        ("Train - Normalized confusion matrix", "true", X_train, y_train),
        ("Validate - Confusion matrix, without normalization", None, X_val, y_val),
        ("Validate - Normalized confusion matrix", "true", X_val, y_val),        
    ]
    
    for title, normalize, X, y in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X,
            y,
            display_labels=model.classes_,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
    
    return
    

def plot_roc_auc(model,X_train, y_train, X_val, y_val):
    """Score Models and return results as a dataframe

    Parameters
    ----------
    model: model
        Model passed into function
    X_train : Numpy Array
        X_train data
    y_train : Numpy Array
        Train target
    X_val : Numpy Array
         X_val data
    y_val : Numpy Array
        Val target        
    
    Returns
    -------
    """
    
    import matplotlib.pyplot as plt

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    
    
    # Plot non-normalized confusion matrix
    titles_options = [
        ("Train - AUC-ROC", X_train, y_train),
        ("Validate - AUC-ROC", X_val, y_val),
    ]
    
    for title, X, y in titles_options:
        
        y_proba = model.predict_proba(X)[:, 1]

        
        roc_auc_score_val = roc_auc_score(y, y_proba)
        roc_auc_score_val = str(format(roc_auc_score_val,".5f"))
        
        title_plus_auc_score = title + ' (' + roc_auc_score_val + ')'
        
        
        fpr,tpr,threshold=roc_curve(y,y_proba)
        plt.figure(figsize=(7,5),dpi=100)
        plt.plot(fpr,tpr,color='green')
        plt.plot([0,1],[0,1],label='baseline',color='red')
        plt.xlabel('FPR',fontsize=15)
        plt.ylabel('TPR',fontsize=15)
        plt.title(title_plus_auc_score,fontsize=20)
        
        plt.show()
    
    return


import seaborn as sns
import matplotlib.pyplot as plt

# Correlation between X and y
def corrs_graph(df_in):
    corr_matrix = df_in.corr()
    fig, ax = plt.subplots(figsize=(16,12))
    ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="Blues")
    plt.show()
    
    
def corrs_X_y(df_in, y):    
    # Find correlations with the Target and sort
    correlations = df_in.corr()[y].sort_values()

    # Display correlations with Target
    print('Most Positive Correlations with Target:')
    print(correlations.dropna().tail(10))

    print()

    print('Most Negative Correlations with Target:') 
    print(correlations.dropna().head(10))
    
    
    
def collinear_col(df_in, n, cols_to_remove=[]):
    # Set the threshold
    threshold = n

    # Empty dictionary to hold correlated variables
    above_threshold_vars = {}

    # Calculate all correlations in dataframe
    correlations = df_in.corr()
    correlations = correlations.sort_values('TARGET_5Yrs', ascending = False)

    # For each column, record the variables that are above the threshold
    for col in correlations:
        above_threshold_vars[col] = list(correlations.index[correlations[col] > threshold])

    # Track columns to remove and columns already examined
    cols_to_remove = []
    cols_seen = []
    cols_to_remove_pair = []

    # Iterate through columns and correlated columns
    for key, value in above_threshold_vars.items():
        # Keep track of columns already examined
        cols_seen.append(key)
        for x in value:
            if x == key:
                next
            else:
                # Only want to remove one in a pair
                if x not in cols_seen:
                    cols_to_remove.append(x)
                    cols_to_remove_pair.append(key)

    cols_to_remove = list(set(cols_to_remove))
    print('Number of columns to remove: ', len(cols_to_remove))
    
    return cols_to_remove
