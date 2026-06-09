from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , classification_report

#Load the Iris Dataset
iris = load_iris()
x,y = iris.data , iris.target

#Important: Scale feature Before KNN
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Split Data
x_test , x_train , y_test , y_train = train_test_split(x_scaled , y , test_size = 0.2 , random_state = 42)

#Create Model and train it
knn = KNeighborsClassifier(n_neighbors= 5)
knn.fit(x_train , y_train)

#Predict and Evaluate
y_pred = knn.predict(x_test)
print('Accuracy : ' , accuracy_score(y_pred , y_test))
print('Classification Report : \n' , classification_report(y_pred , y_test , target_names= iris.target_names))