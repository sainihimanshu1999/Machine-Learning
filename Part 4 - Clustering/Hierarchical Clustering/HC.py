#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset using pandas
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[: , [3,4]].values

#using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x , method = 'ward') )
plt.title('Dendogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean Distance')
plt.show()


#fitting the hierarchial clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)


#visualising the results
plt.scatter(x[y_hc == 0,0], x[y_hc == 0,1], s=100 , c ='red' , label = 'CLuster 1')
plt.scatter(x[y_hc == 1,0], x[y_hc == 1,1], s=100 , c ='yellow' , label = 'CLuster 2')
plt.scatter(x[y_hc == 2,0], x[y_hc == 2,1], s=100 , c ='green' , label = 'CLuster 3')
plt.scatter(x[y_hc == 3,0], x[y_hc == 3,1], s=100 , c ='blue' , label = 'CLuster 4')
plt.scatter(x[y_hc == 4,0], x[y_hc == 4,1], s=100 , c ='magenta' , label = 'CLuster 5')

plt.title('Cluster of clients')
plt.xlabel('Annual income')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

