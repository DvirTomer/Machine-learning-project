import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import plotly.express as px
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("googleplaystore12.csv")
# print(df)


##histograma
fig = px.histogram(df,x='Rating')
fig.show()

# Create Classification version of target variable
df['goodquality'] = ['good' if x >= 4.4 else  'bad' for x in df['Rating']]
# Separate feature variables and target variable
X = df.drop(['Rating','goodquality'], axis = 1)
y = df['goodquality']

# print(df)
# print(y)
print(df['goodquality'].value_counts())


from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)


# print(X_train)
# print(X_test)
# print(y_test)
# print(y_train)

def DecisionTree():
    print("\n**************   DecisionTree   **************")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
    # decision tree
    model1 = DecisionTreeClassifier(random_state=1)
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    ac = model1.score(X_test, y_test)
    print("\nresult: ", ac)
    # print(classification_report(y_test, y_pred1))


def logistic():
    print("\n**************   LogisticRegression   **************")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
    # logistic regression
    model1 = LogisticRegression()
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    ac = model1.score(X_test, y_test)
    print("\nresult: ", ac)
    # print(classification_report(y_test, y_pred1))

def adaboost():
    # adaboost
    print("\n**************   ADABOOST   **************")
    from sklearn.ensemble import AdaBoostClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
    model3 = AdaBoostClassifier()
    model3.fit(X_train, y_train)
    y_pred3 = model3.predict(X_test)
    ac = model3.score(X_test, y_test)
    print("\nresult: ", ac)


def SVM():
    #   SVM
    print("\n**************   SVM   **************")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

    from sklearn.svm import SVC
    ppn = SVC(C=10000, kernel='rbf', degree=3)
    ppn.fit(X_train, y_train)
    y_pred4 = ppn.predict(X_test)
    acc = ppn.score(X_test, y_test)
    print("\nresult: ", acc)
    # print("Accuracy score: " + str(classification_report(y_test, y_pred4)))
    # print("\nConfusion matrix: \n" + str(confusion_matrix(Y_test, y_pred)))
    # print("\nClassification report: \n" + str(classification_report(Y_test, y_pred)))


def KNN():
    # adaboost
    print("\n**************   KNN   **************")
    from sklearn.ensemble import AdaBoostClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
    model3 = KNeighborsClassifier(n_neighbors=4)
    model3.fit(X_train, y_train)
    y_pred5 = model3.predict(X_test)
    ac = model3.score(X_test, y_test)
    print("\nresult: ", ac)

def Naive_Bayes():
    # adaboost
    print("\n**************   Naive_Bayes   **************")
    from sklearn.ensemble import AdaBoostClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
    model3 = clf = GaussianNB()
    model3.fit(X_train, y_train)
    y_pred5 = model3.predict(X_test)
    ac = model3.score(X_test, y_test)
    print("\nresult: ", ac)

DecisionTree()
logistic()
adaboost()
SVM()
KNN()
Naive_Bayes()