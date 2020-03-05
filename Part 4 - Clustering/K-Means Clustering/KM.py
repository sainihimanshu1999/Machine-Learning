#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing datasets with pandas
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

#using the elbow method to find out the number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++' , max_iter =300 , n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of CLuster')
plt.ylabel('wcss')
plt.show()


#applying k-means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++' , max_iter =300 , n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


#visualising the results
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s=100 , c ='red' , label = 'CLuster 1')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s=100 , c ='yellow' , label = 'CLuster 2')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s=100 , c ='green' , label = 'CLuster 3')
plt.scatter(x[y_kmeans == 3,0], x[y_kmeans == 3,1], s=100 , c ='blue' , label = 'CLuster 4')
plt.scatter(x[y_kmeans == 4,0], x[y_kmeans == 4,1], s=100 , c ='magenta' , label = 'CLuster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1] , s=300 , c ='Cyan' , label = 'Centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual income')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

