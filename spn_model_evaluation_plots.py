# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:24:57 2023

@author: japostol
"""
import scikitplot as skplt
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.metrics import roc_curve
import numpy as np



def plot_learning_curves_manual(labels,predictions_all_num,history,class_names,model_name):
    
    
    
    
    plt.rcParams['axes.facecolor'] = 'white'
    # summarize history for accuracy
    plt.plot(history.history['accuracy'], color="darkcyan",linewidth=1)
    
    if 'val_acc' in history.history.keys():
        plt.plot(history.history['val_acc'], color="black",linewidth=1)
    elif 'val_accuracy' in history.history.keys():
        plt.plot(history.history['val_accuracy'],color="black",linewidth=1)

    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accuracy (x100 %)')
    plt.xlabel('Epoch')
    
    if 'val_acc' in history.history.keys():
        plt.legend(['train', 'validation'], loc='upper left')
    if 'val_accuracy' in history.history.keys():
        plt.legend(['train', 'validation'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.grid(False)
    plt.gca().spines['bottom'].set_color('0.5')
    plt.gca().spines['top'].set_color('0.5')
    plt.gca().spines['right'].set_color('0.5')
    plt.gca().spines['left'].set_color('0.5')
    plt.savefig('C:\\Users\\User\\accs_{}.png'.format(model_name), dpi=300)
    plt.show()
    
    
    
    
    
    plt.rcParams['axes.facecolor'] = 'white'
    # summarize history for loss
    plt.plot(history.history['loss'], color="darkcyan",linewidth=1)
    if 'val_loss' in history.history.keys():
        plt.plot(history.history['val_loss'], color="black",linewidth=1)

    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if 'val_loss' in history.history.keys():
        plt.legend(['train', 'validation'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.grid(False)
    plt.gca().spines['bottom'].set_color('0.5')
    plt.gca().spines['top'].set_color('0.5')
    plt.gca().spines['right'].set_color('0.5')
    plt.gca().spines['left'].set_color('0.5')
    plt.savefig('C:\\Users\\User\\losses_{}.png'.format(model_name), dpi=300)
    plt.show()
    
    
    
    # plt.rcParams['axes.facecolor'] = 'white'
    # # roc curve
    # fpr = dict()
    # tpr = dict()
    
    # colors = ['black','red','orange','navy','lime','yellow']
    # for i,item in enumerate( class_names ):
    #     fpr[i], tpr[i], _ = roc_curve(labels[:, i],predictions_all_num[:, i])
    #     plt.plot(fpr[i], tpr[i], lw=1, label='Class: {}'.format(item), color=colors[i])
    #     plt.plot(fpr[i], tpr[i], lw=1, color=colors[i])
    
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.legend(loc="best")
    # plt.title("ROC curve")
    # plt.grid(False)
    # plt.gca().spines['bottom'].set_color('0.5')
    # plt.gca().spines['top'].set_color('0.5')
    # plt.gca().spines['right'].set_color('0.5')
    # plt.gca().spines['left'].set_color('0.5')
    # plt.savefig('C:\\Users\\User\\rocurves_{}.png'.format(model_name), dpi=300)
    # plt.show()



# CROSS-VALIDATION LEARNING CURVE
def plot_learning_curve_scikit(classifier_untrained, Xen, Yen):
    skplt.estimators.plot_learning_curve(classifier_untrained, Xen, Yen,
                                         cv=5, shuffle=False, scoring="accuracy",
                                         n_jobs=-1, figsize=(30,20), title_fontsize=40, text_fontsize=30,
                                         title="Learning Curve")
    
    

# FEATURE IMPORTANCE

def plot_feature_importance(trained_classifier,feature_names):
    
    fig, ax = plt.subplots(figsize=(30, 19))

    skplt.estimators.plot_feature_importances(trained_classifier, feature_names=feature_names,
                                             x_tick_rotation=90, order="descending",
                                             ax=ax, title_fontsize=60, text_fontsize=30,title="Feature Importance")


# ROC CURVE

def plot_roc_scikit(all_predictions_proba,all_true_labels):
    from sklearn.metrics import roc_curve, auc
    # Plot ROC curve
    # Compute ROC curve and AUC
    all_predictions_proba2 = all_predictions_proba[:, 1]
    fpr, tpr, _ = roc_curve(all_true_labels[:,1], all_predictions_proba2)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    
    
        # Add bounding box lines
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1)    
    
    
    # Customize the plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    
    # Add a bounding box


    # Save the plot (adjust the filename as needed)
    #plt.savefig('roc_curve.png', bbox_inches='tight')

    plt.show()
    
    
def plot_roc_scikit_multiple_classifiers(list_of_predictions_proba, list_of_true_labels, list_of_classifier_names):
    from sklearn.metrics import roc_curve, auc

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    for predictions_proba, true_labels, classifier_name in zip(list_of_predictions_proba, list_of_true_labels, list_of_classifier_names):
        fpr, tpr, _ = roc_curve(true_labels, predictions_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{classifier_name} (AUC = {roc_auc:.2f})')

    # Customize the plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curves', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)

    # Save the plot (adjust the filename as needed)
    # plt.savefig('roc_curves.png', bbox_inches='tight')

    plt.show()    
    
    
    
    
# KS Statistic

def plot_ks_statistic(all_true_labels,all_predictions_proba): 
    fig, ax = plt.subplots(figsize=(25, 20), dpi=300)
    skplt.metrics.plot_ks_statistic(all_true_labels, all_predictions_proba, 
                                    title='KS Statistic Plot', ax=ax, figsize=(25, 20), title_fontsize=40, text_fontsize=35)
    # Customize the legend
    legend_labels = ['Benign Class', 'Malignant Class', 'KS Statistic']
    legend = ax.legend(legend_labels, loc='upper right', fontsize=40)
    
    # Customize the line width using ax.plot
    for i in range(len(ax.lines)):
        line = ax.lines[i]  # Assuming there is only one line
        line.set_linewidth(8)
    
    # Show the plot
    plt.show()    
    

    

# RELIABILITY CURVE
def plot_reliability_curve(prediction_list,trues,clf_names):
    
    # TLDR: A calibration curve shows the predicted probabilities of a model 
    # compared to the actual probabilities. This can help you understand whether the model is well-calibrated.

    # A calibration curve is a plot that shows the relationship between 
    # the predicted probabilities and the true positive rate. 
    # The x-axis of the plot represents the predicted probabilities, 
    # and the y-axis represents the true positive rate.

    # A good calibration curve will be close to the line y=x, 
    # which indicates that the predicted probabilities are accurate. 
    # This means that if a classifier predicts that a given instance 
    # has a 70% probability of belonging to the positive class, then about 70% of 
    # instances with predicted probabilities of 70% will actually belong to the positive class.
    
    
    # Plot calibration curve
    fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
    skplt.metrics.plot_calibration_curve(trues,prediction_list,clf_names,ax=ax)

    # Customize the plot
    ax.set_title('Calibration Curve', fontsize=40)
    ax.set_xlabel('Mean Predicted Probability', fontsize=30)
    ax.set_ylabel('Fraction of Positives', fontsize=30)

    # Increase legend size and tick mark sizes
    legend = ax.legend(fontsize=25)
    ax.tick_params(axis='both', which='both', labelsize=20)

    # Customize the plot after the initial plot
    for i in range(len(ax.lines)):
        line = ax.lines[i]  # Assuming there is only one line
        line.set_linewidth(5)  # Set the line width


    # Show the plot
    plt.show()


# COHEN'S KAPPA
def calculate_cohens_kappa_matrices(labels, predictions):
    """
    Calculate Cohen's Kappa score and the observed and expected agreement matrices.

    Parameters:
    - labels: array-like, true labels
    - predictions: array-like, predicted labels

    Returns:
    - kappa_score: float, Cohen's Kappa score
    - observed_agreement: array, observed agreement matrix
    - expected_agreement: array, expected agreement matrix
    """

    # Ensure inputs are numpy arrays
    labels = np.array(labels)
    predictions = np.array(predictions)

    # Calculate observed agreement matrix
    observed_agreement = confusion_matrix(labels, predictions)

    # Calculate expected agreement matrix
    total_samples = len(labels)
    observed_marginals = np.outer(np.sum(observed_agreement, axis=1), np.sum(observed_agreement, axis=0)) / total_samples
    expected_agreement = observed_marginals

    # Calculate Cohen's Kappa score
    kappa_score = cohen_kappa_score(labels, predictions)

    return kappa_score, observed_agreement, expected_agreement






# HEATMAP OF PROBABILITIES VS CLASSES


# LEARNING CURVE (ERROR VS ACCURACY)









#