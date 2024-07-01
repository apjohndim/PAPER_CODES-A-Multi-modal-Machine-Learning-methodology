# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:52:01 2023

@author: japostol
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:38:03 2021

@author: japostol
"""

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def catboost():
    classifier = CatBoostClassifier(depth= 5, iterations= 200, l2_leaf_reg= 3, learning_rate= 0.01)
    return classifier

def xgboost_classifier_scikit_original ():
    # This produces the results in the excel (0.8229% accuracy, ~75% sensitivity)
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(learning_rate= 0.1, max_depth= 5, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100, subsample= 0.8)

def xgboost_classifier_scikit ():
    from sklearn.ensemble import GradientBoostingClassifier
    
    # for GRID-SEARCH based on AUC: 
    # for GRID-SEARCH based on Sensitivity: {'learning_rate': 0.01, 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 1.0}
    # for GRID-SEARCH based on Accuracy: 
    
    return GradientBoostingClassifier(learning_rate= 0.01, max_depth= 7, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 300, subsample= 0.8)


def logistic_regression_classifier():
    return LogisticRegression()


def naive_bayes_classifier():
    return GaussianNB()


def k_nn_classifier():
    return KNeighborsClassifier(n_neighbors= 7, p= 1, weights= 'distance')


def random_forest_classifier():
    
    # GRID-SEARCHED for Recall: 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100
    
    return RandomForestClassifier(max_depth= 5, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100)


def xgboost_classifier():
    return XGBClassifier(colsample_bytree= 1.0, learning_rate= 0.1, max_depth=7, n_estimators=50, subsample=1.0)


def lightgbm_classifier():
    
    # GRID SEARC for Recall: Best Parameters for lightgbm: {'colsample_bytree': 1.0, 'learning_rate': 0.2, 'max_depth': 10, 'n_estimators': 200, 'reg_alpha': 0, 'reg_lambda': 0.1, 'subsample': 0.8}
    
    return LGBMClassifier(colsample_bytree= 1, learning_rate= 0.2, max_depth= 10, n_estimators= 200, reg_alpha= 0, reg_lambda= 0.1, subsample= 0.8)

def svm_classifier():
    # Grid search for recall: Best Parameters for svm: {'C': 0.1, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}
    
    return SVC(probability=True,C= 0.1, degree= 2, gamma= 'scale', kernel= 'linear')


def neural_network_classifier():
    return MLPClassifier()


def adaboost_classifier():
    return AdaBoostClassifier(learning_rate= 0.1, n_estimators= 200)


def gmm_classifier():
    return GaussianMixture()


def lda_classifier():
    return LinearDiscriminantAnalysis()



def selector (classifier_name):
    classifier = None
    if classifier_name == 'catboost':
        classifier = catboost()
    if classifier_name == 'logistic':
        classifier = logistic_regression_classifier()
    if classifier_name == 'bayes':
        classifier = naive_bayes_classifier()
    if classifier_name == 'knn':
        classifier = k_nn_classifier()
    if classifier_name == 'rf':
        classifier = random_forest_classifier()
    if classifier_name == 'xgb':
        classifier = xgboost_classifier_scikit()
    if classifier_name == 'lightgbm':
        classifier = lightgbm_classifier()
    if classifier_name == 'svm':
        classifier = svm_classifier()
    if classifier_name == 'nn':
        classifier = neural_network_classifier()
    if classifier_name == 'adaboost':
        classifier = adaboost_classifier()
    if classifier_name == 'gmm':
        classifier = gmm_classifier()
    if classifier_name == 'lda':
        classifier = lda_classifier()

    return classifier

