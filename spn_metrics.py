# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:24:17 2022

@author: User
"""
import numpy as np
import pandas as pd
from sklearn import metrics

def calculate_metrics(y_true, y_pred, y_pred_binary, positive_class=1):
    
    # Convert labels to arrays if they are not
    y_true = np.array(y_true)
    y_pred_numeric = np.array(y_pred)  # Numeric predictions
    #y_pred_binary = np.where(y_pred_numeric >= 0.5, 1, 0)  # Binary predictions based on threshold 0.5

    # Ensure labels are in 1D format
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.flatten()

    # Convert one-hot encoded labels to binary
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred_binary)

    # True Positives, True Negatives, False Positives, False Negatives
    tp = confusion_matrix[positive_class, positive_class]
    tn = np.sum(confusion_matrix) - np.sum(confusion_matrix[positive_class, :]) - np.sum(confusion_matrix[:, positive_class]) + tp
    fp = np.sum(confusion_matrix[:, positive_class]) - tp
    fn = np.sum(confusion_matrix[positive_class, :]) - tp

    # Classification Metrics (using binary predictions)
    accuracy = metrics.accuracy_score(y_true, y_pred_binary)
    precision = metrics.precision_score(y_true, y_pred_binary, pos_label=positive_class)
    recall = metrics.recall_score(y_true, y_pred_binary, pos_label=positive_class)
    f1 = metrics.f1_score(y_true, y_pred_binary, pos_label=positive_class)

    # Sensitivity (True Positive Rate or Recall)
    sensitivity = recall

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp)

    # Positive Predictive Value (Precision)
    ppv = precision

    # Negative Predictive Value
    npv = tn / (tn + fn)

    # True Positive Rate (Sensitivity or Recall)
    tpr = sensitivity

    # True Negative Rate (Specificity)
    tnr = specificity
    
    # False Positive Rate
    fpr = fp / (fp + tn)
    
    # False Negative Rate\
    fnr = fn / (fn + tp)

    # AUC Score (using numeric predictions)
    auc = metrics.roc_auc_score(y_true, y_pred_numeric[:,1])

    # Cohen's Kappa Score
    kappa = metrics.cohen_kappa_score(y_true, y_pred_binary)

    # Dictionary of Metrics
    metrics_dict = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC Score': auc,
        'Cohen\'s Kappa': kappa,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positive Rate': tpr,
        'True Negative Rate': tnr,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr,
        'Positive Predictive Value': ppv,
        'Negative Predictive Value': npv,
        'Confusion Matrix': confusion_matrix
    }

    # Convert to Pandas DataFrame
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])

    # Separate lists of names and values
    metric_names = metrics_df['Metric'].tolist()
    metric_values = metrics_df['Value'].tolist()

    return metrics_dict, metrics_df, metric_names, metric_values
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    