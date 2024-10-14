# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 23:31:28 2024

@author: mina


APM:
kafi hast dar enteha qutation bezanid va yek ghesmate report bezarid 
ama aval begid ke data chi boode , x ha chi bode y ha chi boode va hadafeton chi boode
va bnegid k har mdoel ch accuracy dashtand va kodom mdoel ba kodom parameter cheghad accuracy bedast ovorde ( daghigh na hodo)
va be onvane porozhe final baram email konid 
moafagh bashid

"""
#-------------Import Libs---------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt


#-------------loading data-------------------------
data=load_breast_cancer()

x=data.data
y=data.target


kf= KFold(n_splits=4,shuffle=True,random_state=42)

#========== (1) LogisticRegression ===============
model= LogisticRegression()
my_params={  }


gs=GridSearchCV(model,param_grid=my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)

gs.best_score_   #np.float64(0.9454717817393874)
gs.best_params_  #{} 


#========== (2) KNeighborsClassifier ===============
model=KNeighborsClassifier()
my_params= { 'n_neighbors':[1,2,4,7,10,14,24],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'] }

gs=GridSearchCV(model,param_grid=my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)

gs.best_score_   #np.float64(0.9437112183591057)
gs.best_params_  #{'metric': 'manhattan', 'n_neighbors': 10}

#========== (3) DecisionTreeClassifier ===============
model= DecisionTreeClassifier()
my_params={'random_state':[42],'max_depth':[4,7,14,21,24,40,74],
           'min_samples_leaf':[2,4,7,14],'min_samples_split':[4,5,7,12]  }

gs=GridSearchCV(model,param_grid=my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)

gs.best_score_    #np.float64(0.9454840933714173)
gs.best_params_   #{'max_depth': 7,'min_samples_leaf': 4,'min_samples_split': 4,'random_state': 42}

#========== (4) RandomForestClassifier ===============
model= RandomForestClassifier()
my_params={'random_state':[42],'n_estimators':[4,7,10,14,24,56,74],'max_features':[4,14,24,74]  }

gs=GridSearchCV(model,param_grid=my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)

gs.best_score_   #np.float64(0.9613537870580123)
gs.best_params_  #{'max_features': 14, 'n_estimators': 56, 'random_state': 42}

#========== (5) SVC ===============
model=SVC()
my_params={ { 'kernel':['linear','poly','rbf'],'degree':[2,3,4],'C':[0.001,0.01,0.1,1,10,100],'gamma':['scale']}}

gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
 
gs.best_score_   #np.float64(0.9578203486654191)
gs.best_params_  #{'C': 1, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}



#================ (plot of the model with best score) ==================


from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(max_features= 14, n_estimators= 56, random_state= 42)
my_params={}
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(model,param_grid=my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)

x_coloum= x[:,2]
x_coloum_2= x[:,3]

y_pred=gs.predict(x_coloum)

plt.scatter(x_coloum,x_coloum_2,c=y,cmap='viridis')
plt.title('Data real')
plt.xlabel('coloumn_3')
plt.ylabel('coloumn_4')
plt.grid()
plt.show()

plt.scatter(x_coloum,x_coloum_2,c=y_pred,cmap='viridis')
plt.title('Data prediction')
plt.xlabel('coloumn_3')
plt.ylabel('coloumn_4')
plt.grid()
plt.show()



#========COMPARISON REPORT======================
'''
FINAL REPORT:
randomforest (RF) model has the best test score (96%)
 with hyperparameters of {'max_features': 14, 'n_estimators': 56, 'random_state': 42}
 so it is the best validation for this data
'''
