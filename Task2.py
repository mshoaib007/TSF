# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:05:45 2021

@author: M Shoaib
"""

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
iris_datasets=datasets.load_iris()
df=pd.DataFrame(iris_datasets.data,columns=iris_datasets.feature_names)
X=df.iloc[:,[0,1,2,3]].values
val=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters = i, init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    val.append(kmeans.inertia_)
plt.plot(range(1,11),val)
plt.title('Using Elbow Method')
plt.xlabel('Total Clusters')
plt.ylabel('WCSS')
plt.show
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300, n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='blue',label='Iris-setosa')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='red',label='Iris-Versicolour')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Iris-Virginica')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label="Centroids")
plt.title("K_means")
plt.legend()
plt.show()