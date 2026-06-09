import numpy as np

#Create 1D Array
a = np.array([10 , 20 , 30 , 40 , 50])
b = np.array([15 , 30 , 45 , 60 , 75])

#Basic Operations
print(a + b)
print(b * 3)
print(np.min(a))
print(np.max(b))
print(np.mean(b))
print(np.std(a))

#Create 2D Array
matrix = np.array([[0 , 20] , [40 , 60] , [80 , 100]])
print(matrix.shape)
print(matrix.T)

