# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 2022

@author: liangyu wang  980025288
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# function that accepts a dataframe as an argument and normalizes all the
# data points in the dataframe
def normalize(data):
    colmin=data.min()
    colmax=data.max()
    d_min=colmin.min()
    d_max=colmax.max()
    data = (data-d_min)/(d_max-d_min)
    return data




# load csv file into a data frame
titanic_liangyu=pd.read_csv("titanic.csv")

# Display the first 3 records.
print(titanic_liangyu.head(3))
print()


# Display the shape of the dataframe
print(titanic_liangyu.shape)
print()


# Display the column names
for col in titanic_liangyu.columns:
    print(col)

print()


# Display the types of columns
print(titanic_liangyu.dtypes)
print()


# Display the counts per column 
for col in titanic_liangyu.columns:
    print(col + ": "+ str(titanic_liangyu[col].count())) 
print()


# Display the missing values per column 
for col in titanic_liangyu.columns:
    print(col + ": "+ str((titanic_liangyu[col].isnull()).sum()))
print()


# print the unique values for columns: (“Sex”, “Pclass”)
print(titanic_liangyu["Sex"].unique())
print(titanic_liangyu["Pclass"].unique())
print()


# generate table of survived versus passenger class
tab=pd.crosstab(titanic_liangyu["Survived"], titanic_liangyu["Pclass"])
print(tab)
print()

# bar chart showing survived versus passenger class
p=tab.plot(kind="bar", stacked=False, rot=0, title="Number of Survived Passengers by Class - Liangyu")
p.legend(title='Classes', loc='upper right')
p.set_ylabel("number of passengers")
print()

# generate table of survived versus passenger gender
tab=pd.crosstab(titanic_liangyu["Survived"], titanic_liangyu["Sex"])
print(tab)

# bar chart showing survived versus passenger gender
p=tab.plot(kind="bar", stacked=False, rot=0, title="Number of Survived Passengers by Gender - Liangyu")
p.legend(title='Gender', loc='upper right')
p.set_ylabel("number of passengers")
print()


#scatter matrix to plot the relationships between the number of survived versus Gender, Passenger class,
#Fare, Number of siblings/spouses aboard, Number of parents/children aboard. 
pd.plotting.scatter_matrix(titanic_liangyu[["Survived", "Sex", "Pclass", "Fare", "SibSp", "Parch"]], 
                           figsize=(10,10))
print()


#Drop the 4 columns identified in point (b.4)
titanic_liangyu=titanic_liangyu.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)


#Transform all the categorical variables into numeric values.  Attach the newly created variables
#to dataframe and drop the original columns
titanic_liangyu=pd.get_dummies(titanic_liangyu)


#Replace the missing values in the Age with the mean of the age.
titanic_liangyu["Age"] = titanic_liangyu["Age"].fillna(titanic_liangyu["Age"].mean())


#Change all column types into float.
for col in titanic_liangyu.columns:
    titanic_liangyu[col]=  titanic_liangyu[col].astype(float)
print(titanic_liangyu.info())
print()


#normalizes all the data points in the dataframe and display the first two records.
print(normalize(titanic_liangyu).head(2))
print()

#plot showing all the variable's histograms.
titanic_liangyu.hist(figsize=(9,10))
print()


#Split features into a dataframe named x_firstname and the target class into another
#dataframe named y_firstname.
x_Liangyu = titanic_liangyu.drop("Survived",axis=1)
y_Liangyu = titanic_liangyu["Survived"]



#Split data into 70% for training and 30% for testing.
x_train_Liangyu, x_test_Liangyu, y_train_Liangyu, y_test_Liangyu  = train_test_split(
    x_Liangyu, y_Liangyu, train_size=0.7, test_size=0.3, random_state=88)


#Using sklearn fit a logistic regression model to the training data.
Liangyu_model=LogisticRegression(random_state=88, max_iter=1000).fit(x_train_Liangyu, y_train_Liangyu)


#Display the coefficients (i.e. the weights of the model). 
print(pd.DataFrame(zip(x_train_Liangyu.columns, np.transpose(Liangyu_model.coef_))))
print()


#Use Sklearn cross_val_score to validate the model on the training data.  Repeat the validation for
#different splits of the train/test. Start at test size 10% and reach test size 50%, increasing test
#sample by 5% at each step.
for i in np.arange (0.10, 0.5, 0.05):
    x_train_Liangyu, x_test_Liangyu, y_train_Liangyu, y_test_Liangyu  = train_test_split(
        x_Liangyu, y_Liangyu, train_size=1-i, test_size=i, random_state=88)
    Liangyu_model=LogisticRegression(random_state=88, max_iter=1000).fit(x_train_Liangyu, y_train_Liangyu)
    scores=cross_val_score(Liangyu_model, x_train_Liangyu, y_train_Liangyu, cv=10)
    print(f"Minimum accuracy score for test size {i*100:.0f}%: ",min(scores))
    print(f"Mean accuracy score for test size {i*100:.0f}%: ", np.mean(scores))
    print(f"Maximum accuracy score for test size {i*100:.0f}%: ",max(scores))
print()

#Rebuild the model using the 70% - 30% train/test split.
x_train_Liangyu, x_test_Liangyu, y_train_Liangyu, y_test_Liangyu  = train_test_split(
    x_Liangyu, y_Liangyu, train_size=0.7, test_size=0.3, random_state=88)
Liangyu_model=LogisticRegression(random_state=88, max_iter=1000).fit(x_train_Liangyu, y_train_Liangyu)


#Define a new variable y_pred_firstname, store the predicted probabilities of the model in this
#variable. 
y_pred_Liangyu= Liangyu_model.predict_proba(x_test_Liangyu)


#Define another variable, name it y_pred_firstname_flag , store in the y_pred_firstname after
#transforming the probabilities into a bolean value of true or false based on a threshold value 
#of 0.5.
y_pred_Liangyu_flag = y_pred_Liangyu[:,1] > 0.5


#Print out the accuracy of the model on the test data.
print(f"Accuracy score (threshold 0.5): {accuracy_score(y_test_Liangyu, y_pred_Liangyu_flag)}")
print()


#Print out the confusion matrix.
print(f"Confusion matrix (threshold 0.5):\n {confusion_matrix(y_test_Liangyu, y_pred_Liangyu_flag)}")
print()


#Print out the classification report.
print(f"Classification Report (threshold 0.5):\n {classification_report(y_test_Liangyu, y_pred_Liangyu_flag)}")
print()



#Repeat steps 3 to 6 with changing the threshold value to 0.75.
y_pred_Liangyu_flag = y_pred_Liangyu[:,1] > 0.75
print(f"Accuracy score (threshold 0.75): {accuracy_score(y_test_Liangyu, y_pred_Liangyu_flag)}")
print()
print(f"Confusion matrix (threshold 0.75):\n {confusion_matrix(y_test_Liangyu, y_pred_Liangyu_flag)}")
print()
print(f"Classification Report (threshold 0.75):\n {classification_report(y_test_Liangyu, y_pred_Liangyu_flag)}")
print()


#Compare the accuracy on the test data with the accuracy generated using the training data.
print(f"Accuracy score for test data: {accuracy_score(y_test_Liangyu, Liangyu_model.predict(x_test_Liangyu))}")
print(f"Accuracy score for training data: {accuracy_score(y_train_Liangyu, Liangyu_model.predict(x_train_Liangyu))}")












