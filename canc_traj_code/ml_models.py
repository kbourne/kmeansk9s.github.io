# Core Libraries for Data Science
import pandas as pd
import numpy as np

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import tree

# pipelines
from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn import set_config # used to display pipeline diagram

# pre-processing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

# Deep learning
import tensorflow as tf

# models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import xgboost as xgb

# Evaluation & metrics
from sklearn import metrics
from sklearn.inspection import permutation_importance

# Data Processing
from data_proc import get_split_data, doggo_data_loading # , getXy, oversample, get_default_dataset, get_minority_dataset, get_future_step_dataset

# operations
from collections import Counter
import time
get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from pathlib import Path
import shutil

def dummy_classify(X, y, test_size=0.3, rand_state=1, avg='weighted'): 

  # Strategies: {“most_frequent”, “prior”, “stratified”, “uniform”, “constant”}, default=”prior”
  # Average: 'weighted'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    prec, recl, accu, f1 = None, None, None, None

    strats = ['most_frequent', 'stratified', 'uniform'] #  'prior', 'constant' - same as m_f in this case

    for strat in strats:
        dummy = DummyClassifier(strategy=strat, random_state=rand_state).fit(X_train, y_train)
        dummy_preds = dummy.predict(X_test)

        prec_score = metrics.precision_score(y_test, dummy_preds)
        recl_score = metrics.recall_score(y_test, dummy_preds)
        accu_score = metrics.accuracy_score(y_test, dummy_preds)
        f1_score = metrics.f1_score(y_test, dummy_preds, average=avg)

        cm = metrics.confusion_matrix(y_test, dummy_preds)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dummy.classes_)

        # Print out results:
        print('Dummy Classifier results - Strategy: ', strat)
        disp.plot()
        plt.show()
        # print('confusion matrix: \n', cm)
        #   # [[TN   FP]
        #   # [ FN   TP]]
        print('precision: ', prec_score, 
              '\nrecall:    ',recl_score, 
              '\naccuracy:  ', accu_score, 
              '\nf1:        ', f1_score, '\n')
  # print("""
  # most_frequent: the predict method always returns the most frequent class 
  # label in the observed y argument passed to fit. The predict_proba method 
  # returns the matching one-hot encoded vector. A high accuracy in this case is
  # indicative of an imbalanced dataset.

  # stratified: the predict_proba method randomly samples one-hot vectors from 
  # a multinomial distribution parametrized by the empirical class prior 
  # probabilities. The predict method returns the class label which got 
  # probability one in the one-hot vector of predict_proba. Each sampled row of 
  # both methods is therefore independent and identically distributed.

  # uniform: generates predictions uniformly at random from the list of unique 
  # classes observed in y, i.e. each class has equal probability.

  # precision - true positives / true positives + false positives 

  # recall - true positives / true positives + false negatives

  # f1 score - harmonic mean of precision and recall
  # """)
  # return prec_score, recl_score, accu_score, f1_score

# use approach in here to set up all models at once?
# https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501
# sets up a pipeline that runs through them all quickly

# Run basic form of all shallow models for initial baseline comparison:
# metrics.classification_report & metrics.confusion_matrix

def run_models(X_train, X_test, y_train, y_test, rand_state=0):

    rnd = 4
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    # scalling the input data
    # Do not need normalization for: Decision Tree, Random Forest, Gradient Boost (DT), XGB
    # Does not make much of difference though, so will leave in, just won't apply going forward
    sc_X = StandardScaler() 
    X_train_ss = sc_X.fit_transform(X_train)
    X_test_ss = sc_X.fit_transform(X_test)
    # not scaling y, because binary 1/0, doesn't need scaling
    # y_train_ss = sc_X.fit_transform(y_train)
    # y_test_ss = sc_X.fit_transform(y_test)
    
    model_pipeline = []
    model_pipeline.append(LogisticRegression(solver='liblinear', random_state=rand_state))
    model_pipeline.append(RidgeClassifier(random_state=rand_state)) # Non-linear SVC - rbf kernel
    model_pipeline.append(SGDClassifier(random_state=rand_state))
    model_pipeline.append(SVC(random_state=rand_state)) # rbf kernel
    model_pipeline.append(LinearSVC(random_state=rand_state)) # linear kernel
    model_pipeline.append(KNeighborsClassifier())
    model_pipeline.append(DecisionTreeClassifier(random_state=rand_state))
    model_pipeline.append(RandomForestClassifier(random_state=rand_state)) # max_depth=10, min_samples_leaf=1, min_samples_split=5, 
    model_pipeline.append(GaussianNB())
    model_pipeline.append(BernoulliNB())
    model_pipeline.append(GradientBoostingClassifier(random_state=rand_state))
    model_pipeline.append(xgb.XGBClassifier(objective="binary:logistic", random_state=rand_state))

    # model_list = ['Logistic Regr','Decision Tree','Random Forest','B-Naive Bayes', 'GradientBoost', 'XGBoostClassf'] # shorter list when needed
    model_list = ['Logistic Regr','Ridge Regress','StochaGD(SGD)','SupVectr(SVC)','LinearKrn-SVC','KNeighbor-KNN','Decision Tree','Random Forest','G-Naive Bayes','B-Naive Bayes', 'GradientBoost', 'XGBoostClassf'] # all
    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    auc_list = []
    cm_list = []
    train_time_list = []

    start_time = time.time()
    elapsed_time = time.time() - start_time
    
    # for model in model_pipeline:
    print('BEGINNING MODEL RUNS')
    
    for i, model in enumerate(model_pipeline):
        start_model_time = time.time()
        
        # don't normalize for the tree models:
        if i in [0,1,2,3,4,5,8,9]:
            # print(f'scale! model #{i}:{model_list[i]}')
            model.fit(X_train_ss, y_train)
        else:
            # print(f'Dont scale! model #{i}:{model_list[i]}')
            model.fit(X_train, y_train)
        
        model_fin_time = time.time() - start_model_time
        y_pred = None
        if i in [0,1,2,3,4,5,8,9]:
            y_pred = model.predict(X_test_ss)
        else: 
            y_pred = model.predict(X_test)
            
        acc_list.append(round(metrics.accuracy_score(y_test, y_pred),rnd))
        pre_list.append(round(metrics.precision_score(y_test, y_pred),rnd))
        rec_list.append(round(metrics.recall_score(y_test, y_pred),rnd))
        f1_list.append(round(metrics.f1_score(y_test, y_pred),rnd))
        fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
        auc_list.append(round(metrics.auc(fpr, tpr),rnd))
        cm_list.append(metrics.confusion_matrix(y_test, y_pred))
            
        train_time_list.append("{:.1f}".format(model_fin_time))
        elapsed_time = time.time() - start_time
        
        print(f"Step {i+1} of {len(model_list)}: {model_list[i]} computation complete - elapsed time: {elapsed_time:.3f}, model fit time: {model_fin_time:.3f} AUC score: {metrics.auc(fpr, tpr)}")

    fig = plt.figure(figsize = (16,36))
    for i in range(len(cm_list)):
        cm = cm_list[i]
        model = model_list[i]
        sub = fig.add_subplot(6, 2, i+1).set_title(model)
        cm_plot = sns.heatmap(cm, annot=True, cmap = 'Blues_r')
        cm_plot.set_xlabel('Predicted Values')
        cm_plot.set_ylabel('Actual Values ')

    result_df = pd.DataFrame({'Model':model_list, 'Accuracy': acc_list, 
                                'Precision':pre_list, 'Recall':rec_list, 
                                'F1 Score':f1_list, 'AUC':auc_list, 'Train Time':train_time_list})
  
    return result_df


# Optimized Model Functions
# this is the structure in the docs, but have to do the proba calc outside of it?  How to do that in pipeline?
def prec_recall_auc(y_test, y_pred_proba):
    # calculate the precision-recall auc
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_proba)
    auc_score = metrics.auc(recall, precision)
    print('PR AUC: %.3f' % auc_score)
    return auc_score

def evaluate(model, X_test, y_test):
    
    t0 = time.perf_counter()
    
    test_preds = model.predict(X_test)
    
    t1 = time.perf_counter()
    print(f"Predict time to compute: {t1-t0:0.3f} seconds")
    
    test_preds_proba = model.predict_proba(X_test)[:, 1]
    
    t2 = time.perf_counter()
    print(f"Test predict probas time to compute: {t2-t1:0.3f} seconds")
    
    test_accu = metrics.accuracy_score(y_test, test_preds)
    test_prec = metrics.precision_score(y_test, test_preds)
    test_recl = metrics.recall_score(y_test, test_preds)
    test_f1sc = metrics.f1_score(y_test, test_preds)
    test_rauc = metrics.roc_auc_score(y_test, test_preds_proba) # test_preds_proba[:,1])
    test_raup = prec_recall_auc(y_test, test_preds_proba) 
    
    # y should be a 1d array, got an array of shape (38525, 2) instead.
    test_confusion_matrix = metrics.confusion_matrix(y_test, test_preds)

    print("-"*50)
    print("-"*50)
    print(f'Test Accuracy :       {test_accu:.5f}')
    print(f'Test Precision score: {test_prec:.5f}')
    print(f'Test Recall score:    {test_recl:.5f}')
    print(f'Test F1 score:        {test_f1sc:.5f}')
    print(f'Test ROC_AUC score:   {test_rauc:.5f}')
    print(f'Test PRC_AUC score:   {test_raup:.5f}')
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    
def evaluate_rr(model, X_test, y_test):
    
    t0 = time.perf_counter()
    
    test_preds = model.predict(X_test)
    
    t1 = time.perf_counter()
    print(f"Predict time to compute: {t1-t0:0.3f} seconds")
    
    # use decision function instead of pred_proba for Ridge Classifier
    test_preds_proba = model.decision_function(X_test) # [:, 1]
    
    t2 = time.perf_counter()
    print(f"Test predict probas time to compute: {t2-t1:0.3f} seconds")
    
    test_accu = metrics.accuracy_score(y_test, test_preds)
    test_prec = metrics.precision_score(y_test, test_preds)
    test_recl = metrics.recall_score(y_test, test_preds)
    test_f1sc = metrics.f1_score(y_test, test_preds)
    test_rauc = metrics.roc_auc_score(y_test, test_preds_proba) # test_preds_proba[:,1])
    test_raup = prec_recall_auc(y_test, test_preds_proba) 
    
    # y should be a 1d array, got an array of shape (38525, 2) instead.
    test_confusion_matrix = metrics.confusion_matrix(y_test, test_preds)

    print("-"*50)
    print("-"*50)
    print(f'Test Accuracy :       {test_accu:.5f}')
    print(f'Test Precision score: {test_prec:.5f}')
    print(f'Test Recall score:    {test_recl:.5f}')
    print(f'Test F1 score:        {test_f1sc:.5f}')
    print(f'Test ROC_AUC score:   {test_rauc:.5f}')
    print(f'Test PRC_AUC score:   {test_raup:.5f}')
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    
def make_confusion(model, X_vals, y_vals, dataset_type):
    y_preds = model.predict(X_vals)
    cm = metrics.confusion_matrix(y_vals, y_preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        
    # Print out results:
    print('Confusion Matrix results, Dataset: ', dataset_type)
    disp.plot()
    plt.show()
    
def get_influence(model, X_train):
    return pd.DataFrame({'Variable':X_train.columns,
              'Importance':model.feature_importances_}).sort_values('Importance', ascending=False)