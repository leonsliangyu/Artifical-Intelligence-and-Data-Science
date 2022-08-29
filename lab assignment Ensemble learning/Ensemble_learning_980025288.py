# -*- coding: utf-8 -*-
#liangyu wang
#980025288

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import seaborn as sns; sns.set()



#load data into pandas dataframe
df_liangyu=pd.read_csv('pima-indians-diabetes.csv')

#Add the column names
df_liangyu.columns =['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi',
                     'age', 'class']

#Check the names and types of columns.
print(df_liangyu.info())
print()

#Check the missing values.
for col in df_liangyu.columns:
    print(col + ": "+ str((df_liangyu[col].isnull()).sum()))
print()


#Check the statistics of the numeric fields (mean, min, max, median, count,..etc.)
print(df_liangyu.describe().applymap('{:,.2f}'.format))
print("medians:")
print(df_liangyu.median())
print()


#There is no categorical feature
print(df_liangyu.select_dtypes(exclude=["int64", "float64"]).columns.tolist())


#Print out the total number of instances in each clas
print(df_liangyu["class"].value_counts())
print()


#Prepare a standard scaler transformer to transform all the numeric values.
transformer_liangyu = StandardScaler()


#Split the features from the class.
features = df_liangyu.drop("class", axis=1)
target = df_liangyu["class"]


#Split your data into train 70% train and 30% test, use 42 for the seed
X_train_liangyu, X_test_liangyu, y_train_liangyu, y_test_liangyu = train_test_split(
    features, target, train_size=0.7, test_size=0.3,
    random_state=42)


X_train_scaled = transformer_liangyu.fit_transform(X_train_liangyu)
X_test_scaled = transformer_liangyu.fit_transform(X_test_liangyu)

##Hard voting 
print("Hard Voting\n")
#Define 5 classifiers 
lr_W = LogisticRegression( max_iter=1400)
rf_W = RandomForestClassifier()
svc_W= SVC()
dt_W = DecisionTreeClassifier(criterion="entropy", max_depth =42)
et_W= ExtraTreesClassifier()


#Define a voting classifier that contains all the above classifiers
#as estimators.
voting_clf = VotingClassifier(
    estimators=[('lr', lr_W), ('rf', rf_W), ('svc', svc_W),
    ('dt', dt_W), ('et', et_W)],
    voting='hard')


#Fit the training data to the voting classifier and predict the first
#three instances of test data.  Print out for each classifier and for
#each instance the predicted and the actual values.
for i in range(0, 3):
    print('Predictions: ')
    for clf in (lr_W, rf_W, svc_W, dt_W, et_W, voting_clf):
        clf.fit(X_train_scaled, y_train_liangyu)
        y_pred = clf.predict(X_test_scaled[i,:].reshape(1,-1))
        print(clf.__class__.__name__, y_pred[0])
    print('\nActual: ', y_test_liangyu.iloc[i])
    print('\n\n')
    

##Soft voting
print("Soft Voting\n")

svc_W= SVC(probability=True)

#Define a voting classifier that contains all the above classifiers
#as estimators.
voting_clf = VotingClassifier(
    estimators=[('lr', lr_W), ('rf', rf_W), ('svc', svc_W),
    ('dt', dt_W), ('et', et_W)],
    voting='soft')


#Fit the training data to the voting classifier and predict the first
#three instances of test data.  Print out for each classifier and for
#each instance the predicted and the actual values.
for i in range(0, 3):
    print('Predictions: ')
    for clf in (lr_W, rf_W, svc_W, dt_W, et_W, voting_clf):
        clf.fit(X_train_scaled, y_train_liangyu)
        y_pred = clf.predict_proba(X_test_scaled[i,:].reshape(1,-1))
        print(clf.__class__.__name__, y_pred[0])
    print('\nActual: ', y_test_liangyu.iloc[i])
    print('\n\n')


## Random forests & Extra Trees


#pipeline 1 with Extra Trees Classifier 
pipeline1_liangyu= Pipeline(steps=[
                ('StandardScaler', transformer_liangyu),
                ('extree', et_W)
                ])

#pipeline 2 with Decision Tree Classifier 
pipeline2_liangyu= Pipeline(steps=[
                ('StandardScaler', transformer_liangyu),
                ('extree', dt_W)
                ])


#Fit the original data to both pipelines.
pipeline1_liangyu.fit(X_train_liangyu, y_train_liangyu)
pipeline2_liangyu.fit(X_train_liangyu, y_train_liangyu)


#Carry out a 10 fold cross validation for both pipelines
score1=cross_val_score(pipeline1_liangyu,  X_train_liangyu, y_train_liangyu,
                       cv=KFold(n_splits=10, random_state=42, shuffle=True))
score2=cross_val_score(pipeline2_liangyu,  X_train_liangyu, y_train_liangyu,
                       cv=KFold(n_splits=10, random_state=42, shuffle=True))


#Print out the ten scores and the mean of the ten scores 
print("Pipeline 1 10-fold cross validation mean score: ", score1.mean())
print("Pipeline 2 10-fold cross validation mean score: ", score2.mean())
print()

#Predict the test using both pipelines and printout the confusion matrix,
#precision, recall and accuracy scores
pipelines=[pipeline1_liangyu, pipeline2_liangyu]

for i in range(0,2):
    y_pred= pipelines[i].predict(X_test_liangyu)
    accuracy=accuracy_score(y_test_liangyu, y_pred)
    print("accuracy: ", accuracy)
    precision=precision_score(y_test_liangyu, y_pred)
    print("precision: ", precision)
    recall=recall_score(y_test_liangyu, y_pred)
    print("recall: ", recall)
    confusion_mat = confusion_matrix(y_test_liangyu, y_pred)
    print('\nConfusion matrix:\n',confusion_mat)
    print()



##Extra Trees and Grid search

##efine the grid search parameters
param_grid_88 = {
               'extree__n_estimators': list(range(10, 3000, 20)),
               'extree__max_depth': list(range(1, 1000, 2))
               }

#Create grid search object
grid_search_88 =RandomizedSearchCV(estimator = pipeline1_liangyu,
                               param_distributions = param_grid_88,
                               verbose=3, 
                               n_iter =30,
                               cv=3,
                               refit = True, 
                               random_state=88,
                               )

#Carryout a randomized grid search on the first Pipeline
grid_search_88.fit(X_train_liangyu, y_train_liangyu)


#Print out the best parameters
print("\nBest parameters: ")
print(grid_search_88.best_params_)
print()


#Printout the best estimator model score
print("Best score: ")
grid_score=grid_search_88.best_score_
print(grid_score)
print()


#Use the fine-tuned model identified during the randomized
#grid search i.e the best estimator saved in the randomized
#grid search object to predict the test data
best_model = grid_search_88.best_estimator_
y_grid_pred= best_model.predict(X_test_liangyu)
print(y_grid_pred)

#Printout the precision, re_call and accuracy
report=classification_report(y_test_liangyu, y_grid_pred, digits=4)
print("\nClassification Report:\n", report)




