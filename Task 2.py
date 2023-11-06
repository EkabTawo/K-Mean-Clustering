

#This is the K-means Clustering Algorithm in python sklearn that is used in tackling clustering
#related issues.


#Code to impoort the necessary libraries. 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting

#Code for sklearn package in machine learning in python:
from sklearn.cluster import KMeans, MeanShift
from sklearn.datasets import make_blobs

#Chunk of code that reads tthe dataset from a CSV file. Then assigns the values of a chosen row(variables)
#to another variable in python and then graph a scattered plot of the values.
df = pd.read_csv('country_data.csv')

X = df.iloc[:,[1,3]].values

plt.scatter(X[:,0], X[:,1], c = 'red', s = 50)


plt.show()

#Chunk of code to chose the number of clusters I want and then fit their shapes. 

kmeans = KMeans(n_clusters = 2)
kmeans.fit(X)

#Chunk of code that plots the dataset values asigned as well as locates and show the centers.

main, tab  = plt.subplots()
tab.scatter(X[:,0], X[:, 1], c = kmeans.predict(X), s = 50, cmap = 'plasma')

centers = kmeans.cluster_centers_
print('Centroids:', centers)

for i in range(centers.shape[0]):
    tab.text(centers[i,0], centers[i, 1], str(i), c = 'blue',
            bbox=dict(boxstyle="round", facecolor='green', edgecolor='black'))
    

tab.set_xlabel('child_mort')
tab.set_ylabel('health')
