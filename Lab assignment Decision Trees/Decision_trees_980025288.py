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
from matplotlib import pyplot as plt
import graphviz 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import joblib

pd.set_option('display.max_columns', None)


#load data into pandas dataframe
df_liangyu=pd.read_csv('student-por.csv', sep=';')


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
print(df_liangyu.select_dtypes(include=["int64"]).median())
print()


#Check the categorical values.
for col in df_liangyu.columns:
    if(df_liangyu[col].dtypes == object):
        print(col, end="   ")
        print(df_liangyu[col].unique())
        print(df_liangyu[col].value_counts())



#Create a new target variable.
pass_liangyu=np.where(df_liangyu['G1']+df_liangyu['G2']+df_liangyu['G3']>=35, 1, 0)
df_liangyu['pass_liangyu'] = pass_liangyu.tolist()

print('\n')


#Drop the columns G1, G2, G3 permanently.
df_liangyu.drop(['G1','G2','G3'], axis=1, inplace=True)


#Separate the features from the target variable (class)
Features_liangyu = df_liangyu.drop("pass_liangyu", axis=1)
target_variable_liangyu = df_liangyu["pass_liangyu"]


#Print out the total number of instances in each class
print(target_variable_liangyu.value_counts())
print()

#Create two lists one to save the names of numeric fields and one
#to save the names of categorical fields
numeric_features_liangyu = Features_liangyu.select_dtypes(include=["int64"]).columns.tolist()
cat_features_liangyu = Features_liangyu.select_dtypes(include=["object"]).columns.tolist()


#Prepare a column transformer to handle all the categorical variables
#and convert them into numeric values using one-hot encoding.
transformer_liangyu = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(),  cat_features_liangyu),
    ], remainder='passthrough'
)


#Prepare a classifier decision tree model
clf_liangyu= DecisionTreeClassifier(criterion="entropy", max_depth = 5)


#Build a pipeline. The pipeline have two steps the first the column transformer,
#second the DecisionTree model.
pipeline_liangyu = Pipeline(steps=[
                ('onehotenc', transformer_liangyu),
                ('decisiontree', clf_liangyu)
                ])


#Split data into 80% training and 20% testing.
X_train_liangyu, X_test_liangyu, y_train_liangyu, y_test_liangyu = train_test_split(
    Features_liangyu, target_variable_liangyu, train_size=0.8, test_size=0.2,
    random_state=88)


#Fit the training data to the pipeline
pipeline_liangyu.fit(X_train_liangyu, y_train_liangyu)


#Cross validate the output on the training data using 10-fold cross validation
scores=cross_val_score(pipeline_liangyu, X_train_liangyu, y_train_liangyu,
                      cv=KFold(n_splits=10, shuffle=True, random_state=88) )

#Print out the ten scores and the mean of the ten scores 
print("10-fold cross validation scores: ", scores)
print()
print("Mean score: ", scores.mean())


plt.figure(figsize=(35,16))
tree.plot_tree(clf_liangyu,filled = True, fontsize=10)

#Visualize the tree using Graphviz.
dot_data= tree.export_graphviz(clf_liangyu,
                     out_file=None,
                     feature_names = 
                     pipeline_liangyu.named_steps['onehotenc'].get_feature_names_out().tolist(), 
                     class_names= ['1', '0'],
                     filled = True)

graph = graphviz.Source(dot_data) 
graph.format = "png"
graph.render("student-por")
graph 


#Two accuracy score one for the model on the training set 
#i.e. X_train, y_train and the other on the testing set
# i.e. X_test, y_test
accuracy=  pipeline_liangyu.score(X_train_liangyu ,y_train_liangyu)
print("Training set accuracy score: ", accuracy)
accuracy2=  pipeline_liangyu.score(X_test_liangyu ,y_test_liangyu)
print("Testing set accuracy score: ", accuracy2)
print()

#Use the model to predict the test data and printout the accuracy,
#precision and recall scores and the confusion matrix
y_train_pred= pipeline_liangyu.predict(X_train_liangyu)

accuracy=accuracy_score(y_train_liangyu, y_train_pred)
print("accuracy: ")
print(accuracy)
print()

precision=precision_score(y_train_liangyu, y_train_pred)
print("precision: ")
print(precision)
print()
recall=recall_score(y_train_liangyu, y_train_pred)
print("recall: ")
print(recall)
print()
confusion_mat = confusion_matrix(y_train_liangyu, y_train_pred)
print('\nConfusion matrix:\n',confusion_mat)
print()



#Define the grid search parameters
parameters = {'decisiontree__min_samples_split' : range(10,300,20),
              'decisiontree__max_depth': range(1,30,2),
              'decisiontree__min_samples_leaf':range(1,15,3)
              }


#Create a grid search object
grid_search_liangyu= RandomizedSearchCV(estimator= pipeline_liangyu, 
                                  scoring='accuracy',
                                  param_distributions=parameters,
                                  cv=5,
                                  n_iter = 7,
                                  refit = True,
                                  verbose = 3
                                  )

grid_search_liangyu.fit(X_train_liangyu, y_train_liangyu)


#Print out the best parameters
print("\nBest parameters: ")
print(grid_search_liangyu.best_params_)
print()

#Printout the accuracy score
print("Score: ")
print(grid_search_liangyu.score(X_train_liangyu, y_train_liangyu))
print()

#Print out the best estimator 
print("Best estimator: ")
best_model= grid_search_liangyu.best_estimator_
print(best_model)
print()


#Fit the test data using the fine-tuned model identified during
#grid search 
best_model.fit(X_test_liangyu, y_test_liangyu)


#Use the best estimator to predict the test data and printout
#the precision, recall and accuracy
y_best_pred= grid_search.predict(X_test_liangyu)


accuracy=accuracy_score(y_test_liangyu, y_best_pred)
print("accuracy: ")
print(accuracy)
print()

precision=precision_score(y_test_liangyu, y_best_pred)
print("precision: ")
print(precision)
print()
recall=recall_score(y_test_liangyu, y_best_pred)
print("recall: ")
print(recall)
print()


#Save the model using the joblib
joblib.dump(best_model, "best_model_liangyu.pkl")

#Save the full pipeline using the joblib
joblib.dump(pipeline_liangyu, 'pipeline_liangyu.pkl')




