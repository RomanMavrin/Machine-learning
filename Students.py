import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
x = np.array([[1],[2],[3],[4],[5], [6]])
y = np.array([50, 55, 65, 70, 78, 85])
model = LinearRegression()
model.fit(x,y)
hour = 4.5
predict_mark = model.predict([[hour]])
print(predict_mark)
plt.scatter(x,y,color = 'blue', label = 'Исходные данные')
plt.plot(x,model.predict(x),color='green', label = 'Модель')
plt.scatter(hour, predict_mark, color = 'black', s = 100, label = 'Предсказание')
plt.xlabel('Часы подготовки')
plt.ylabel('Оценка')
plt.title('Модель линейной регрессии')
plt.legend()
plt.grid(True)
plt.show()