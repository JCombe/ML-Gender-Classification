# Author Julian Biscombe
# Date 18-03-2018

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

### Data ###
# Data for training
# [height, weight, shoe_size]
X = [[190, 86, 10],
     [183, 78, 11],
     [178, 70, 10],
     [180, 68, 9],
     [178, 76, 10],
     [178, 83, 10],
     [175, 68, 10],
     [188, 84, 5],
     [173, 77, 9],
     [173, 70, 10],
     [180, 76, 38],
     [183, 83, 12],
     [178, 72, 10],
     [178, 80, 46],
     [175, 65, 10],
     [183, 73, 40]]
X += [[165, 54, 7],
      [165, 56, 8],
      [160, 60, 7],
      [180, 58, 9],
      [163, 48, 6],
      [170, 52, 8],
      [160, 58, 8],
      [163, 52, 7],
      [168, 54, 9],
      [168, 54, 8],
      [170, 59, 9],
      [155, 51, 6],
      [155, 54, 8],
      [158, 52, 7],
      [160, 52, 8],
      [157, 49, 7],
      [178, 64, 10],
      [160, 54, 7],
      [173, 55, 8],
      [152, 49, 7],
      [173, 57, 8],
      [168, 53, 32],
      [168, 57, 7],
      [160, 60, 8]]

Y = ['male'] * 16 + ['female'] * 24

# Data for testing
# [height, weight, shoe_size]
X_test=[
        [188, 70, 38],[189, 94, 34],[180, 70, 42],[159, 61, 37],[165, 58, 39],
        [162, 54, 34],[171, 60, 40],[170, 70, 40],[143, 45, 37],[153, 48, 39]
        ]

Y_test=['male','male','male','female','male',
        'female','male','male','female','female']


#DecTree
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X, Y)
clf_predict_tree = clf_tree.predict(X_test)
print("Predict DTree :", clf_predict_tree)


#KNeighbors
clf_kneighbors = KNeighborsClassifier()
clf_kneighbors.fit(X,Y)
clf_predict_neighbors = clf_kneighbors.predict(X_test)
print("Predict Neighbors :",clf_predict_neighbors)

#GaussianNB
clf_gaus = GaussianNB()
clf_gaus.fit(X, Y)
clf_predict_gaus = clf_gaus.predict(X_test)
print("Predict Gaussian :", clf_predict_gaus)

#RandomForest
clf_ranfor = RandomForestClassifier()
clf_ranfor.fit(X,Y)
clf_predict_ranfor = clf_ranfor.predict(X_test)
print("Predict Random Forest :",clf_predict_ranfor)

# Summary: the classifiers KNeigbors and Gaussian naive bayes where the most accurate ones.

