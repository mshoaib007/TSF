# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 14:05:59 2021

@author: M Shoaib
"""
import pandas as pd

from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import pydotplus
import graphviz
import numpy 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_class']
iris_df = pd.read_csv('Iris.csv', names = column_names, header = None)
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
X = numpy.delete(X,(0),axis=0)
y = iris_df['iris_class'].values
y = numpy.delete(y,(0),axis=0)
#Create an instance of the decision tree classifier
dec_tree = DecisionTreeClassifier(criterion="entropy")

#Fit the model with the training data X_train and y_train
dec_tree.fit(X, y)
#initialize a StringIO class
dot_data = StringIO()

#file name to save the image
filename = "iris_flower_tree.png"

feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] #input values
class_names = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica'] #output values

#convert the decision tree model into dot data
out = export_graphviz(dec_tree,
                      feature_names=feature_names,
                      out_file=dot_data,
                      class_names=class_names,
                      filled=True,
                      special_characters=True,
                      rotate=False)

#convert the dot data into a graph
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

#save the graph
graph.write_png(filename)

#open and plot the graph
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest');