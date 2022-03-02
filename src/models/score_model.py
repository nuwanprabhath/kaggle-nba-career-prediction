
def score_model(X_train, y_train, X_val, y_val, X_test, y_test, y_base):
    model_scores = []
    best_acc = 0
    i = 0
    
    if y_base 
    
    
    for name, model in models.items():
        i = i+1;
        clf = model
        if dump_model == "YES":
            job.dump(clf, "../models/williams_sean-week2_" + name + ".joblib", compress=3)
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
    return best_clf



    



