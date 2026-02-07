import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# X =np.array([
#     [2,5],
#     [3,4],
#     [4,3],
#     [5,2],
#     [6,1],
#     [7,0]
#
# ])
#
# y = [50, 55, 65, 75, 85, 90]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
#
# model = LinearRegression()
# model.fit(X_train, y_train)
#
#
# y_pred = model.predict(X_test)
#
#
# print('MAE: ', mean_absolute_error(y_test, y_pred))
# print('R2: ', r2_score(y_test, y_pred))
# print("Коэффициенты ", model.coef_)
# print("Смещение ", model.intercept_)
#
#
# plt.scatter(X_train[:, 0], y_train, color='blue', label='Обучающие данные')
# plt.scatter(X_test[:, 0], y_test, color='red', label='Тестовые данные')
#
# # Генерируем линию для предсказания
# X_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
# # Второй признак фиксируем как 0 (или можно взять среднее — но здесь логично 0)
# X_line_full = np.column_stack([X_line, np.zeros_like(X_line)])
#
# y_line = model.predict(X_line_full)
#
# plt.plot(X_line, y_line, color='green', label='Предсказание')
# plt.xlabel('Часы')
# plt.ylabel('Оценка')
# plt.title('Линейная регрессия (по первому признаку)')
# plt.legend()
# plt.grid(True)
# plt.show()



'''ДЗ
Нужно предсказать количество строк кода в день по двум факторам: 1. часы кодинга, 2. количество отвлечений
Проанализировать коэффициенты
Написать словами: на сколько строк кода влияет +1 час кодинга
и на сколько стрк кода влияет +1 отвлечение'''

# X = np.array([
#     [1, 8],
#     [2, 6],
#     [3, 5],
#     [4, 4],
#     [5, 3],
#     [6, 2],
#     [7, 1],
#     [8, 0]
# ])

# y = np.array([50, 90, 120, 160, 190, 230, 260, 300])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)


# model = LinearRegression()
# model.fit(X_train,y_train)

# y_pred = model.predict(X_test)

# print('MAE: ', mean_absolute_error(y_test, y_pred))
# print('R2: ', r2_score(y_test, y_pred))
# print("Коэффициенты ", model.coef_)
# print("Смещение ", model.intercept_)
# print(f"На сколько строк кода влияет +1 час кодинга: {model.coef_[0]:.0f}")
# print(f"На сколько строк кода влияет +1 отвлечение: {model.coef_[1]:.0f}")


# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.scatter(X_train[:, 0], y_train, color='blue', label='Обучающие данные')
# plt.scatter(X_test[:, 0], y_test, color='red', label='Тестовые данные')

# X_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
# X_line_full = np.column_stack([X_line, np.zeros_like(X_line)])
# y_line = model.predict(X_line_full)

# plt.plot(X_line, y_line, color='green', label='Предсказание')
# plt.xlabel('Часы')
# plt.ylabel('Количество строк кода')
# plt.title('Линейная регрессия (по первому признаку)')
# plt.legend()
# plt.grid(True)

# # График по второму признаку (отвлечения)
# plt.subplot(1, 2, 2)
# plt.scatter(X_train[:, 1], y_train, color='blue', label='Обучающие данные')
# plt.scatter(X_test[:, 1], y_test, color='red', label='Тестовые данные')

# X_line2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
# avg_hours = X[:, 0].mean()
# X_line2_full = np.column_stack([np.full_like(X_line2, avg_hours), X_line2])
# y_line2 = model.predict(X_line2_full)

# plt.plot(X_line2, y_line2, color='green', label='Предсказание')
# plt.xlabel('Отвлечения')
# plt.ylabel('Строки кода')
# plt.title('Зависимость от количества отвлечений')
# plt.legend()
# plt.grid(True)
# plt.show()

from sklearn.linear_model  import LogisticRegression
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.naive_bayes   import GaussianNB
from sklearn.tree  import DecisionTreeClassifier
from sklearn.svm  import SVC

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
