from sklearn.metrics import accuracy_score,classification_report, roc_auc_score

def eval_metrics(y, y_pred):
    print('Accuracy Score: ', accuracy_score(y, y_pred))
    print('Classification Report:')
    print(classification_report(y, y_pred))
    print('ROC_AUC_Score: ' , roc_auc_score(y, y_pred))