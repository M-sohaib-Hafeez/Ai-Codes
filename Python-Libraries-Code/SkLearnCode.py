import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report , ConfusionMatrixDisplay

#Lod Data
data = load_iris()
x , y = data.data , data.target

#Split Data
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state = 42)

#Create and Train Model
model = KNeighborsClassifier(n_neighbors =  5)
model.fit(x_train , y_train)

#Make Prediction
ypred = model.predict(x_test)

#Evaluate
print('Accuracy : ' , accuracy_score(ypred,y_test))
print('Classification Report : ' ,classification_report(ypred , y_test , target_names = data.target_names))

#Plot
ConfusionMatrixDisplay.from_predictions(y_test , ypred , display_labels = data.target_names)
plt.show()





