#!/usr/bin/env python
# coding: utf-8

# In[185]:


# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


# In[186]:


col_names = ['para1', 'para2', 'para3', 'para4', 'para5', 'para6', 'para7', 'para8', 'para9','para10','para11','para12','para13','para14','para15','para16','target']
# load dataset
# Function importing Dataset 
def importdata(): 
    balance_data = pd.read_csv("pendigitstraining.csv", header=None, names=col_names) 
    # Printing the dataswet shape 
    print ("Dataset Length: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data


# In[187]:


def splitdataset(balance_data):
    # Seperating the target variable 
    feature_cols = ['para1', 'para2', 'para3', 'para4', 'para5', 'para6', 'para7', 'para8', 'para9','para10','para11','para12','para13','para14','para15','para16'] 
    X = balance_data[feature_cols]
    Y = balance_data.target
    
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 


# In[188]:


def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=8, min_samples_leaf=10) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 


# In[189]:


# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 8, min_samples_leaf = 10) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 


# In[190]:


# Function to make test predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
# Function to make predictions 
def predictiont(y_train, clf_object): 
  
    # Predicton on test with giniIndex 
    x_pred = clf_object.predict(y_train) 
    print("Predicted values:") 
    print(x_pred) 
    return x_pred 


# In[191]:


# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy Test: ", 
    accuracy_score(y_test,y_pred)*100) 
       
    print("Report : ", 
    classification_report(y_test, y_pred)) 
def cal_accuracyt(X_train,x_pred):
    print ("Accuracy Train : ", 
    accuracy_score(X_train,x_pred)*100)
    
     


# In[192]:


# Driver code 
def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    
    print("Training Entropy:") 
    # Prediction using entropy 
    X_pred_entropy = predictiont (X_train, clf_entropy) 
    cal_accuracyt(X_train,x_pred)
        

      
      
# Calling main function 
if __name__=="__main__": 
    main()


# In[ ]:




