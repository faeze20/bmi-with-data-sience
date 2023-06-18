import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans





female_data=pd.read_csv("female_data.csv")
female_data["F2_329"].fillna(value=female_data["F2_329"].mean(), inplace=True)
female_data["F2_330"].fillna(value=female_data["F2_330"].max(), inplace=True)
female_data["F2_332"].fillna(value=female_data["F2_332"].max(), inplace=True)
female_data["F2_357"].fillna(value=female_data["F2_357"].max(), inplace=True)


print(female_data.isna().sum())

x = female_data.values[:, :-1]
y = female_data.values[:, -1]


kmeans=KMeans(n_clusters=5, random_state=0)
kmeans.fit(x)
labels=kmeans.labels_
print(labels)
plt.scatter(x[:,0],x[:,1],c="y")
plt.show()
