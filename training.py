#importation des paquets

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
# Chargement des données
data=load_iris()

#recuperation des features

X=data.data

#recuperation target

y=data.target

#repartition du dataset

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.33, random_state=42)

#Training
knn=KNeighborsClassifier(n_neighbors=3)
model=knn.fit(X_train,y_train)

#score
score=model.score(X_test,y_test) #0,98

#enregistrement
dump(model,'classifier.joblib')