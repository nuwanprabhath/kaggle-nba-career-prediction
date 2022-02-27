import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import joblib as job
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve

def score_base(y_train_preds, y_train, y_val_preds, y_val, f1_average='weighted'):
    name = 'Base'
    model_scores = []
    t_acc = accuracy_score(y_train, y_train_preds)
    t_prec = precision_score(y_train, y_train_preds)
    t_rec = recall_score(y_train, y_train_preds)
    t_f1 = f1_score(y_train, y_train_preds, average=f1_average)
    #t_auc = roc_auc_score(y_t, clf.predict_proba(X_t)[:, 1])
    v_acc = accuracy_score(y_val, y_val_preds)
    v_prec = precision_score(y_val, y_val_preds)
    v_rec = recall_score(y_val, y_val_preds)
    v_f1 = f1_score(y_val, y_val_preds, average=f1_average)
    #v_auc = roc_auc_score(y_v, clf.predict_proba(X_v)[:, 1])
    model_scores.append([name, t_acc, t_prec, t_rec, t_f1, v_acc, v_prec, v_rec, v_f1])
    df_model_scores = pd.DataFrame (model_scores, columns = ['model','t_accuracy','t_precision','t_recall','t_F1','v_accuracy','v_precision','v_recall','v_F1'])
    display(df_model_scores)

def fit_score_models(models, X_t, y_t, X_v, y_v, show_plots="NO", dump_model="NO"):
    model_scores = []
    best_acc = 0
    i = 0
    for name, model in models.items():
        i = i+1;
        clf = model
        if dump_model == "YES":
            job.dump(clf, "../models/williams_sean-week3_" + name + ".joblib", compress=3)
        clf.fit(X_t, y_t)
        t_preds = clf.predict(X_t)
        t_acc = accuracy_score(y_t, t_preds)
        if i == 1:
            best_acc = t_acc
            best_clf = clf
        else:
            if t_acc > best_acc:
                best_acc = t_acc
                best_clf = clf            
        t_prec = precision_score(y_t, t_preds)
        t_rec = recall_score(y_t, t_preds)
        t_f1 = f1_score(y_t, t_preds)
        t_auc = roc_auc_score(y_t, clf.predict_proba(X_t)[:, 1])
        v_preds = clf.predict(X_v)
        v_acc = accuracy_score(y_v, v_preds)
        v_prec = precision_score(y_v, v_preds)
        v_rec = recall_score(y_v, v_preds)
        v_f1 = f1_score(y_v, v_preds)
        v_auc = roc_auc_score(y_v, clf.predict_proba(X_v)[:, 1])
        model_scores.append([name, t_acc, t_prec, t_rec, t_f1, t_auc, v_acc, v_prec, v_rec, v_f1, v_auc])
    df_model_scores = pd.DataFrame (model_scores, columns = ['model','t_accuracy','t_precision','t_recall','t_F1','t_auc','v_accuracy','v_precision','v_recall','v_F1','v_auc'])
    display(df_model_scores)
    if show_plots == "YES": 
        ConfusionMatrixDisplay.from_estimator(clf, X_t, y_t, display_labels=clf.classes_, cmap=plt.cm.Blues)
        ConfusionMatrixDisplay.from_estimator(clf, X_v, y_v, display_labels=clf.classes_, cmap=plt.cm.Blues)
        fpr,tpr,threshold=roc_curve(y_t, t_preds)
        plt.figure(figsize=(7,5),dpi=100)
        plt.plot(fpr,tpr,color='green')
        plt.plot([0,1],[0,1],label='baseline',color='red')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()
        fpr,tpr,threshold=roc_curve(y_v, v_preds)
        plt.figure(figsize=(7,5),dpi=100)
        plt.plot(fpr,tpr,color='green')
        plt.plot([0,1],[0,1],label='baseline',color='red')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()

    return best_clf