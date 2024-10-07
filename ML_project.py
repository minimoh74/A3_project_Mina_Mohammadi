# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 23:31:28 2024

@author: mina
"""

from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()

x=data.data
y=data.target

from sklearn.model_selection import KFold
kf= KFold(n_splits=4,shuffle=True,random_state=42)

#========== (1) LogisticRegression ===============
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
my_params={
    }

from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)

gs.best_score_   #np.float64(0.9454717817393874)
gs.best_params_  #{} 

#========== (2) KNeighborsClassifier ===============
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
my_params= { 'n_neighbors':[1,2,4,7,10,14,24],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'] }

from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)

gs.best_score_   #np.float64(0.9437112183591057)
gs.best_params_  #{'metric': 'manhattan', 'n_neighbors': 10}
