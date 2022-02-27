# Source: AdvDSI-Lab2-Exercise1-Solutions.ipynb
# Author: Anthony So

def print_reg_perf(X, y, set_name=None, model=None):
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
    
    # display(df_model_scores)
    
    return df_model_scores
    
def score_models(X_train, y_train, X_val, y_val, y_base, includeBase, model):
    
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
    
   
    df_model_scores = print_reg_perf(X_train, y_train, set_name='Train', model=model)
    
    df_model_scores_val = print_reg_perf(X_val, y_val, set_name='Validate', model=model)
    
    df_model_scores = pd.concat([df_model_scores,df_model_scores_val],ignore_index = True, axis=0)
    
    display(df_model_scores)
    
    #if includeBase == True:
    #   print_reg_perf(y_train, y_base, set_name='Base', model=model)
    
    return

    
    
    
    
