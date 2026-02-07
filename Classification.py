from sklearn.linear_model  import LogisticRegression
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.naive_bayes   import GaussianNB
from sklearn.tree  import DecisionTreeClassifier
from sklearn.svm  import SVC
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# logreg_clf = LogisticRegression

# logreg_clf.fit(features', labels)
# logreg_clf.predict(test_features)

data = pd.read_csv('iris.csv')
print(data.head(5))

data.drop('Id', axis =1, inplace=True)

X = data.iloc [:, :-1].values
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=27)
print(X_train)
print('='*50)
print(y_train)
# X = np.array([
#     [1,6]
#     [2,5]
#     [3,4]
#     [4,3]
#     [5,2]
#     [6,1]
#     [7,0]
#     [8,0]
# ])

SVC_model = SVC()

KNN_model = KNeighborsClassifier(n_neighbors=5)

SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)

SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = SVC_model.predict(X_test)

print(accuracy_score(SVC_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))

print(confusion_matrix(SVC_prediction, y_test))
print(classification_report(KNN_prediction, y_test))
