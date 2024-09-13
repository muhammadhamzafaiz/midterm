# Qestion no: 03 Prediction of Price Va
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

cake = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
price = np.array([100, 150, 250, 300, 400, 450, 600, 800, 1000, 1500, 0, 0])

model = LinearRegression()
model.fit(cake[:10], price[:10])

predicted_prices = model.predict(cake)
price[10:] = predicted_prices[10:]

plt.scatter(cake, price, color='blue')
plt.plot(cake, predicted_prices, color='red')
plt.xlabel('Cake (in pounds)')
plt.ylabel('Price (in dollars)')
plt.title('Cake Price Prediction')
plt.show()