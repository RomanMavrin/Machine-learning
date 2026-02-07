import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
x = np.array([[1],[2],[3],[4],[5]])
y = np.array([60, 75, 90, 110, 130])
model = LinearRegression()
model.fit(x,y)
seniority = 3.5
predict_salary = model.predict([[seniority]])
print(predict_salary)
plt.scatter(x,y,color = 'blue', label = 'Исходные данные')
plt.plot(x,model.predict(x),color='red', label = 'Модель')
plt.scatter(seniority  , predict_salary, color = 'green', s = 100, label = 'Предсказание')
plt.xlabel('Опыт')
plt.ylabel('Зарплата')
plt.title('Простая модель линейной регрессии')
plt.legend()
plt.grid(True)
plt.show()