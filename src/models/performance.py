# Source: AdvDSI-Lab2-Exercise1-Solutions.ipynb
# Author: Anthony So

def print_reg_perf(y_preds, y_actuals, set_name=None):
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
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    
    print(f"Accuracy      {set_name}: { accuracy_score(y_actuals, y_preds)}")
    print(f"RMSE          {set_name}: {mse(y_actuals, y_preds, squared=False)}")
    print(f"MAE           {set_name}: {mae(y_actuals, y_preds)}")
    print(f'Precision     {set_name}: {precision_score(y_actuals, y_preds)}')
    print(f'Recall        {set_name}: {recall_score(y_actuals, y_preds)}')
    print(f'F1            {set_name}: {f1_score(y_actuals, y_preds)}')
