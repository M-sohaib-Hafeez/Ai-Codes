import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix , roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report

#Load Dataset
data = load_breast_cancer()
x,y = data.data,data.target

#split Data
x_train , x_test , y_train , y_test = train_test_split (x,y , test_size= 0.2 , random_state= 42)

#Create Model
model = LogisticRegression(max_iter = 10000)
model.fit(x_train , y_train)

#Make Predictions
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)

#Evaluate
print('Accuracy : ' , accuracy_score(y_test , y_pred))
print('Classification Report : \n' , classification_report(y_test , y_pred))

#plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
auc = roc_auc_score(y_test, y_proba[:, 1])

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
