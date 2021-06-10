from altair.vegalite.v4.schema.core import Data
from multiapp import MultiApp
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from models import NB, SVM, LR
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return(accuracy)

def app():
    st.header('Modelling')
    st.subheader('Choose the target variable')
    
    with open('./pickles/scaled.pk', 'rb') as f:
        data = pk.load(f)
        cols = tuple(data.columns)
    
    target = st.selectbox('', cols)

    inde = [i for i in cols if i != target]

    X, y = np.array(data[inde]).reshape(data.shape[0], -1), np.array(data[target]).reshape(data.shape[0], )

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state = 1, test_size = 0.3)

    st.subheader('Select models to be trained')
    models = st.multiselect('', ('Naive Bayes', 'SVM-Linear', 'Logistic Regression'))

    model = []

    for i in models:
        if i == 'Naive Bayes':
            nb = NB.NaiveBayes()
            nb.fit(train_x, train_y)
            pred = nb.predict(test_x)
            test_acc = accuracy(test_y, pred)

            #model.append(('Naive Bayes From Scratch', [test_acc, cohen_kappa_score(test_y, pred), roc_auc_score(test_y, pred)]))

            mnb = MultinomialNB()
            mnb.fit(train_x, train_y)
            pred_mnb = mnb.predict(test_x)
            acc_mnb = accuracy(test_y, pred_mnb)

            model.append(('Naive Bayes',
                         [acc_mnb, cohen_kappa_score(test_y, pred_mnb), roc_auc_score(test_y, pred_mnb)]))
        
        if i == 'SVM-Linear':
            svm = SVM.SVM()
            svm.fit(train_x, train_y)
            pred = svm.predict(test_x)
            test_acc = accuracy(test_y, pred)

            model.append(('SVM', [test_acc, cohen_kappa_score(test_y, pred), roc_auc_score(test_y, pred)]))

            svc = LinearSVC()
            svc.fit(train_x, train_y)
            pred_svc = svc.predict(test_x)
            acc_svc = accuracy(test_y, pred_svc)

            #model.append(('SVM', [acc_svc, cohen_kappa_score(test_y, pred_svc), roc_auc_score(test_y, pred_svc)]))

        if i == 'Logistic Regression':
            lr = LR.LogisticRegression()
            lr.fit(train_x, train_y)
            pred = lr.predict(test_x)
            acc = accuracy(test_y, pred)

            #model.append(('Logistic Regression From Scratch', [acc, cohen_kappa_score(test_y, pred), roc_auc_score(test_y, pred)]))

            lr_s = LogisticRegression()
            lr_s.fit(train_x, train_y)
            pred_s = lr_s.predict(test_x)
            acc_s = accuracy(test_y, pred_s)
            
            model.append(('Logistic Regression', [acc_s, cohen_kappa_score(test_y, pred_s), roc_auc_score(test_y, pred_s)]))
            
    acc_dic = dict(model)
    acc_df = pd.DataFrame(acc_dic).T
    acc_df.columns = ['Accuracy Score', "Cohen's Kappa", 'ROC AUC Score']

    st.subheader('Performance of each model compared with that of Scikit Learn')
    st.dataframe(acc_df)
