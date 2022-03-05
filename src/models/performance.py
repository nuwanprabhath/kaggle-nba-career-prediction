# Source: AdvDSI-Lab2-Exercise1-Solutions.ipynb
# Author: Anthony So

def score_model(X, y, set_name=None, model=None):
    """Print regular performance statistics for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    import pandas as pd
    import numpy as np
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    
    
    
    y_preds = model.predict(X)
    y_predict_proba = model.predict_proba(X)[:, 1]
   
    perf_accuracy  = accuracy_score(y, y_preds)
    perf_mse       = mse(y, y_preds, squared=False)
    perf_mae       = mae(y, y_preds)
    perf_precision = precision_score(y, y_preds)
    perf_recall    = recall_score(y, y_preds)
    perf_F1        = f1_score(y, y_preds)
    perf_AUC       = roc_auc_score(y, y_predict_proba)

    # print(f'ROC-AUC       {set_name}: { roc_auc_score(y_preds, model.predict_proba(y_preds)[:, 1])}')
    
    
    model_scores = []
    
    model_provided = model
    
    model_scores.append([set_name, perf_accuracy, perf_mse, perf_mae, perf_precision, perf_recall, perf_F1,perf_AUC])
    
    df_model_scores = pd.DataFrame (model_scores, columns = ['Set Name','ACC','MSE','MAE','PREC','RECALL','F1','AUC'])
    
    return df_model_scores

# New NULL Model

def score_null_model(y_train, y_base, set_name = None):
    """Print regular performance statistics for the provided data

    Parameters
    ----------
    y_train : Numpy Array
        Predicted target
    y_base : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed
    model : str
        Model to be used

    Returns
    -------
    """
    import pandas as pd
    import numpy as np
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    
    perf_accuracy  = accuracy_score(y_base, y_train)
    perf_mse       = mse(y_base, y_train, squared=False)
    perf_mae       = mae(y_base, y_train)
    perf_precision = precision_score(y_base, y_train)
    perf_recall    = recall_score(y_base, y_train)
    perf_F1        = f1_score(y_base, y_train)
    perf_AUC       = None #roc_auc_score(y_base, y_predict_proba)

    # print(f'ROC-AUC       {set_name}: { roc_auc_score(y_preds, model.predict_proba(y_preds)[:, 1])}')
    
    
    model_scores = []
    
    model_scores.append([set_name, perf_accuracy, perf_mse, perf_mae, perf_precision, perf_recall, perf_F1,perf_AUC])
    
    df_model_scores = pd.DataFrame (model_scores, columns = ['Set Name','ACC','MSE','MAE','PREC','RECALL','F1','AUC'])
    
    return df_model_scores


def score_models(X_train = None, y_train = None, X_val = None, y_val = None, y_base = None, includeBase = False, model = None):
    
    """Score Models and return results as a dataframe

    Parameters
    ----------
    X_train : Numpy Array
        X_train data
    y_train : Numpy Array
        Train target
    X_val : Numpy Array
         X_val data
    y_val : Numpy Array
        Val target        
    includeBase: Boolean
        Calculate and display baseline
    model: model
        Model passed into function
    
    Returns
    -------
    """
    
    import pandas as pd
    import numpy as np
    
    df_model_scores = pd.DataFrame()
    
    if includeBase == True:
        df_model_scores_base = score_null_model(y_train = y_train, y_base = y_base, set_name='Base')
        
        df_model_scores = pd.concat([df_model_scores,df_model_scores_base],ignore_index = True, axis=0)
    
    if X_train.size > 0:
        df_model_scores_train = score_model(X_train, y_train, set_name='Train', model=model)
        
        df_model_scores = pd.concat([df_model_scores,df_model_scores_train],ignore_index = True, axis=0)
    
    if X_val.size > 0:
        df_model_scores_val = score_model(X_val, y_val, set_name='Validate', model=model)
        
        df_model_scores = pd.concat([df_model_scores,df_model_scores_val],ignore_index = True, axis=0)
    
    display(df_model_scores)
    
    return

def score_models2(X_train = None, y_train = None, X_val = None, y_val = None, X_test = None, y_test = None, y_base = None, includeBase = False, model = None):
    
    """Score Models and return results as a dataframe

    Parameters
    ----------
    X_train : Numpy Array
        X_train data
    y_train : Numpy Array
        Train target
    X_val : Numpy Array
         X_val data
    y_val : Numpy Array
        Val target
    X_test : Numpy Array
         X_test data
    y_test : Numpy Array
        Test target
    includeBase: Boolean
        Calculate and display baseline
    model: model
        Model passed into function

    Returns
    -------
    """

    import pandas as pd
    import numpy as np

    df_model_scores = pd.DataFrame()

    if includeBase == True:
        df_model_scores_base = score_null_model(y_train = y_train, y_base = y_base, set_name='Base')

        df_model_scores = pd.concat([df_model_scores,df_model_scores_base],ignore_index = True, axis=0)

    if X_train.size > 0:
        df_model_scores_train = score_model(X_train, y_train, set_name='Train', model=model)

        df_model_scores = pd.concat([df_model_scores,df_model_scores_train],ignore_index = True, axis=0)

    if X_val.size > 0:
        df_model_scores_val = score_model(X_val, y_val, set_name='Validate', model=model)

        df_model_scores = pd.concat([df_model_scores,df_model_scores_val],ignore_index = True, axis=0)

    if X_test.size > 0:
        df_model_scores_test = score_model(X_test, y_test, set_name='Test', model=model)

        df_model_scores = pd.concat([df_model_scores,df_model_scores_test],ignore_index = True, axis=0)

    display(df_model_scores)

    return
