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

#========== (3) DecisionTreeClassifier ===============
from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier()
my_params={'random_state':[42],'max_depth':[4,7,14,21,24,40,74],
           'min_samples_leaf':[2,4,7,14],'min_samples_split':[4,5,7,12]
    }

from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)

gs.best_score_    #np.float64(0.9454840933714173)
gs.best_params_   #{'max_depth': 7,'min_samples_leaf': 4,'min_samples_split': 4,'random_state': 42}

#========== (4) RandomForestClassifier ===============
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier()
my_params={'random_state':[42],'n_estimators':[4,7,10,14,24,56,74],'max_features':[4,14,24,74]
    }

from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)

gs.best_score_   #np.float64(0.9613537870580123)
gs.best_params_  #{'max_features': 14, 'n_estimators': 56, 'random_state': 42}

#========== (5) SVC ===============
from sklearn.svm import SVC
model=SVC()
my_params={ 'kernel':['linear','poly','rbf'],'C':[1,474],'gamma':[0.004,0.07]}

from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
 
gs.best_score_
gs.best_params_ 
