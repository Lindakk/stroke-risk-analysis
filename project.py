# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:51:46 2022

@author: Linda Deckerli

This is for my final project:use multiple machine learning models to perform classifications of stroke=1 or stroke=0
"""

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


#---------------Section 1: preparing and analyzing the data-------------
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# check the null values for each column
df.isnull().sum()
df.info()


# Since the column 'bmi' has 201 nulls, so I decided to fill na to its mean value
df['bmi'] = df['bmi'].fillna(round(df['bmi'].mean(),2))

# check the basci matrices of this dataset
df=df.drop('id',axis=1)
df.describe()

# check unique values for each column
df['gender'].unique()
df['ever_married'].unique()
df['smoking_status'].unique()
df['work_type'].unique()
df['Residence_type'].unique()
df['smoking_status'].unique()

# check how many 'Other' cells in 'gender' column
df.loc[df['gender']=='Male'].count()
df.loc[df['gender']=='Female'].count()
df.loc[df['gender']=='Other'].count()

# Since there is only one cell marked as 'Other' for gender, I am going to assign it to a 'Male'.
df.index[df['gender']=='Other']
df.at[3116, 'gender'] = 'Male'

# Assign Female=1, Male=0
df['gender']=df['gender'].map({'Female':1,'Male':0})

# Assign Yes=1, No=0 in 'ever_married' column
df['ever_married']=df['ever_married'].map({'Yes':1,'No':0})


# Assign 'Never_worked'=0, everything else=1 ('Private', Govt_job,'children', 'Self-employed') in 'work_type' column
def f(df):
    if df['work_type']=='Never_worked':
        val=0 
    else:     
        val=1
    return val 
df['work_type']=df.apply(f,axis=1)


#Assign 'never smoked' and 'Unknow'=0, everything else=1 ('smokes' and 'formerly smoked')in 'smoking_status' column
def f(df):
    if df['smoking_status']=='never smoked':
        val=0
    elif df['smoking_status']=='Unknown':
         val=0
    else:
        val=1
    return val 
df['smoking_status']=df.apply(f,axis=1)

#Assign 'Urban'=1, 'Rural'=0 in 'Residence_type' column
df['Residence_type']=df['Residence_type'].map({'Urban':1,'Rural':0})


# check the feature correlation with the lable class 'stroke'
corr = df.corr()
corr = corr['stroke'].sort_values().iloc[:-1]
plt.matshow(corr)
plt.figure(figsize=(10,4),dpi=200)
plt.title('feature correlation with stroke')
sns.barplot(x=corr.index, y=corr.values)
plt.xticks(rotation=90)

# check the relationships between age and avg_glucose_level
sns.scatterplot(x='age',y='avg_glucose_level',hue='stroke',data=df)

# check if the lable column is balanced
df['stroke'].value_counts().plot.bar() 
plt.xlabel('stroke')
plt.ylabel('counts')

#--------------Section 2: try different machine learning models with imbalanced lable class-------------
 
X = df.drop(['stroke'],axis=1)
y= df['stroke']

#Split the df table into X_train, x_test, y_test and y_train, 50/50
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5, random_state=3)


#--------KNN MODEL-----------------------------------------
#scale the data
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# set k=1-10
accuracy_rates = []
for k in np.arange(1,11):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(scaled_X_train,y_train)
    y_pred = knn_model.predict(scaled_X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_rates.append(accuracy)

#Print out and plot the accuracy rates 
print(accuracy_rates)
plt.plot(np.arange(1,11),accuracy_rates)
plt.ylabel('accuracy rate')
plt.xlabel('K Neighbors')

#k=10 is the optimal value, and compute the performance measures
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(scaled_X_train,y_train)
y_pred_test = knn_model.predict(scaled_X_test)

# Compute the accuracy, MAE and RMSE when K=10
accuracy = accuracy_score(y_test,y_pred_test)
mae = mean_absolute_error(y_test,y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred_test))
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

# compute and plot the confusion matrix values
plot_confusion_matrix(knn_model,X_test,y_test)

cm = confusion_matrix(y_test, y_pred_test)
TP=cm[0][0]
FN=cm[0][1]
FP=cm[1][0]
TN=cm[1][1]
TPR = TP/(TP + FN)
TNR = TN/(TN + FP) 
k10Table=pd.DataFrame({'TP':TP,'FP':FP,'TN':TN,'FN':FN,'accuracy':accuracy,'TPR':TPR,'TNR':TNR},index=[0])

#----------------Logistic Regression--------------------
log_model = LogisticRegression()
log_model.fit(scaled_X_train,y_train)
y_pred_log = log_model.predict(scaled_X_test)

# Compute the accuracy,MAE and RMSE of the prediction
accuracy = accuracy_score(y_test,y_pred_log)
mae = mean_absolute_error(y_test,y_pred_log)
rmse = np.sqrt(mean_squared_error(y_test,y_pred_log))
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

# compute and plot the confusion matrix values
plot_confusion_matrix(log_model,scaled_X_test,y_test)

# Compute the performance measures and create a table to hold these values
cm_log = confusion_matrix(y_test, y_pred_log )
TP=cm_log[0][0]
FN=cm_log[0][1]
FP=cm_log[1][0]
TN=cm_log[1][1]
TPR = TP/(TP + FN)
TNR = TN/(TN + FP) 
table_log=pd.DataFrame({'TP':TP,'FP':FP,'TN':TN,'FN':FN,'accuracy':accuracy,'TPR':TPR,'TNR':TNR},index=[0])

#------------------------Random Forest------------------
rfc_model = RandomForestClassifier(random_state=3)
rfc_model.fit(scaled_X_train, y_train)
y_pred_rfc = rfc_model.predict(scaled_X_test)

# Compute the accuracy,MAE and RMSE of the prediction
accuracy = accuracy_score(y_test,y_pred_rfc)
mae = mean_absolute_error(y_test,y_pred_rfc)
rmse = np.sqrt(mean_squared_error(y_test,y_pred_rfc))
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

#compute and plot the confusion matrix
plot_confusion_matrix(rfc_model,scaled_X_test,y_test)
 
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
TP=cm_rfc[0][0]
FN=cm_rfc[0][1]
FP=cm_rfc[1][0]
TN=cm_rfc[1][1]
TPR = TP/(TP + FN)
TNR = TN/(TN + FP) 
table_rfc=pd.DataFrame({'TP':TP,'FP':FP,'TN':TN,'FN':FN,'accuracy':accuracy,'TPR':TPR,'TNR':TNR},index=[0])


#-------------------Section 3: try different machine learning models with balanced lable class values-------------

#----Using SMOTE class to balance the training data------------
oversample = SMOTE(random_state=3)
X_train_balanced, y_train_balanced = oversample.fit_resample(scaled_X_train, y_train)

# check and plot the training dataset
y_train_balanced.value_counts().plot.bar(color=['blue','red']) 
plt.xlabel('Stroke')
plt.ylabel('counts')

# create a function to run different maching learning models
def run_model(model,X_train_balanced,scaled_X_test,y_train_balanced,y_test):
    model.fit(X_train_balanced,y_train_balanced)
    preds = model.predict(scaled_X_test)
    accuracy = accuracy_score(y_test,preds)
    mae = mean_absolute_error(y_test,preds)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    
    plot_confusion_matrix(model,scaled_X_test,y_test)
    
    cm = confusion_matrix(y_test, preds)
    TP=cm[0][0]
    FN=cm[0][1]
    FP=cm[1][0]
    TN=cm[1][1]
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP) 
    table=pd.DataFrame({'TP':TP,'FP':FP,'TN':TN,'FN':FN,'accuracy':accuracy,'TPR':TPR,'TNR':TNR},index=[0])
    print(table)
   
   
# -----------try KNN(K=10)-------------
model_knn=KNeighborsClassifier(n_neighbors=10)
run_model(model_knn,X_train_balanced,scaled_X_test,y_train_balanced,y_test)

# ---logistic regression----------
model_log = log_model = LogisticRegression()
run_model(model_log,X_train_balanced,scaled_X_test,y_train_balanced,y_test)

#---------Random Froest------------
model_rf = RandomForestClassifier(random_state=3)
run_model(model_rf,X_train_balanced,scaled_X_test,y_train_balanced,y_test)

# --------------Additional attempts---------------
#----------- stratifiedKFold with Random Froest------------
skf = StratifiedKFold(n_splits=10,random_state=3,shuffle=True)

# perform SMOTE within each fold
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    sm = SMOTE()
    X_train_oversampled, y_train_oversampled = sm.fit_sample(X_train, y_train)
    
    model = RandomForestClassifier(random_state=3)
    model.fit(X_train_oversampled, y_train_oversampled )  
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    TP=cm[0][0]
    FN=cm[0][1]
    FP=cm[1][0]
    TN=cm[1][1]
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP) 
    table=pd.DataFrame({'TP':TP,'FP':FP,'TN':TN,'FN':FN,'accuracy':accuracy,'TPR':TPR,'TNR':TNR},index=[0])
    print(table)
    
    
#----SVC model with class_weight='balanced'--------------------
#Split the df table into X_train, x_test, y_test and y_train, 50/50
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5, random_state=3)
#scale the data
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

svc= SVC(class_weight='balanced')

#create a grid search to find the best parameters
param_grid = {'C':[1,5,10],'kernel':['linear','rbf'],'gamma':['auto','scale']}

grid = GridSearchCV(svc, param_grid)
grid.fit(scaled_X_train,y_train)

grid_preds = grid.predict(scaled_X_test)
plot_confusion_matrix(grid,scaled_X_test, y_test)
cm = confusion_matrix(y_test, grid_preds)
TP=cm[0][0]
FN=cm[0][1]
FP=cm[1][0]
TN=cm[1][1]
TPR = TP/(TP + FN)
TNR = TN/(TN + FP) 
table=pd.DataFrame({'TP':TP,'FP':FP,'TN':TN,'FN':FN,'accuracy':accuracy,'TPR':TPR,'TNR':TNR},index=[0])
print(table)
