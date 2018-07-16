# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 21:28:58 2018

@author: Erman
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import collections
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from numpy import array

class Survey():
    
    
    def __init__(self):
      self.df=pd.read_csv("survey_data.csv")
      self.labels=[]
      
      self.scalerDict={}
      self.modelDict={}
      self.typeRelationDict={}
      self.rawX=[]
      self.rawY=[]
      self.testX=[]
      self.testY=[]
      self.trainX=[]
      self.trainY=[]
      self.testOutputs=[]
      self.clf = RandomForestClassifier(bootstrap=True,random_state=42)
      
    def produceOutputs(self):
                        
               
        self.rawY=self.df['overall_customer_satisfaction'].values.tolist()
        
        df2=self.df.drop('overall_customer_satisfaction', axis=1)
        self.rawX=df2.values.tolist()
        
    def encodeDelayed(self):
        le = preprocessing.LabelEncoder()
        delayeds=self.df['was_flight_delayed'].values.tolist()
        le.fit(delayeds)
        delayeds=le.transform(delayeds)
        self.df['was_flight_delayed']=delayeds
        
    def imputeDelays(self):

        
        delayedMinutes_Corrected=[]
        
        for ii in range(len(self.df['delay_minutes'])):
            if pd.isnull(self.df['delay_minutes'][ii]):
                delayedMinutes_Corrected.append(0)
            else:
                delayedMinutes_Corrected.append(self.df['delay_minutes'][ii])
        
#        print(len(delayedMinutes_Corrected) )       
                
        delayHandling_Corrected=[]
        
        for ii in range(len(self.df['delay_handling'])):
            if pd.isnull(self.df['delay_handling'][ii]):
                delayHandling_Corrected.append(5)
            else:
                delayHandling_Corrected.append(self.df['delay_handling'][ii])
         
        
        self.df['delay_minutes']=delayedMinutes_Corrected
        self.df['delay_handling']=delayHandling_Corrected
        
    def setTestSets(self, ratio):
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.rawX, self.rawY, test_size=ratio, random_state=42)
    

    def trainModel(self):
        self.clf.fit(self.trainX,self.trainY)
        
    def predictAndScore(self):
        y_pred=self.clf.predict(self.testX)
        print("Accuracy Score: ", accuracy_score(y_pred,self.testY ))
        print("Confusion Matrix: ")
        print( confusion_matrix(y_pred,self.testY ))
        print("Classification Report: ")
        print( classification_report(y_pred,self.testY ))
        
    def getImportances(self):
        forest=self.clf
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
        X=array(self.trainX)
        print("Feature ranking:")

        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()    

    def scaleNStore(self):
         
         scaler = StandardScaler()
         scaler.fit(self.rawX)
         self.trainX=scaler.transform(self.trainX)
         self.testX=scaler.transform(self.testX)
           
                   
        
if __name__ == '__main__':
    
    testRatio=0.33
    mySurvey=Survey()
    mySurvey.encodeDelayed()
    mySurvey.imputeDelays()
    mySurvey.produceOutputs()
    mySurvey.setTestSets(testRatio)
    mySurvey.scaleNStore()
    
    
    mySurvey.trainModel()
    mySurvey.predictAndScore()
    mySurvey.getImportances()