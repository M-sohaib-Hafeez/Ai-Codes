import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([10,20,30,40,50])

plt.plot(x,y , color = 'yellow' , label = 'line')
plt.title('Simple Line Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

plt.bar(['Sohaib' , 'Maham' , 'Fahad'] , [21 , 19 , 23] , color = 'green')
plt.title('Simple Bar Plot')
plt.show()

plt.scatter(x,y)
plt.title('Simple Scatter Plot')
plt.show()

