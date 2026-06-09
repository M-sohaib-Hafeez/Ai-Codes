from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

#load data
data = load_iris()
x = data.data

#create model and train
model = KMeans(n_clusters = 3 , random_state = 42)
model.fit(x)

#Evaluate
labels = model.labels_
print('Cluster label : ' , labels)
print('Inertia (WCSS)' , model.inertia_)
print('Silhouette Score : ' , silhouette_score(x , labels))

