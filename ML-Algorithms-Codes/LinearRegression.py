import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
  
# Generate some sample data : house size (sq ft) vs price ($)
x = np.array([[600],[800],[1000],[1200],[1400],[1600],[1800],[2000]])
y = np.array([150000,200000,250000,300000,350000,400000,450000,500000])

# split into Training and Test set
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(x_train,y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)


# Evaluate
print('Slope (m):', model.coef_[0])
print('Intercept (b):', model.intercept_)
print('R2 Score:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))

# Plot
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.legend()
plt.show()