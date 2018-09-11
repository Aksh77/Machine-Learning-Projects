from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
# Split data for training and testing
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

#Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
# Initialize classifier
gnb = GaussianNB()
# Train classifier
model_gnb = gnb.fit(train, train_labels)
# Make predictions
preds_gnb = gnb.predict(test)
# Evaluate accuracy
acc_gnb = accuracy_score(test_labels, preds_gnb)
print("Accuracy of Gaussian Naive Bayes Classifier- ", acc_gnb)

#K-Nearest Neighbours classifier
from sklearn.neighbors import KNeighborsClassifier
acc_knn = []
neigh = []
for i in range(1,101):
    # Initialize classifier
    knn = KNeighborsClassifier(n_neighbors = i)
    # Train classifier
    model_knn = knn.fit(train, train_labels)
    # Make predictions
    preds_knn = knn.predict(test)
    # Evaluate accuracy
    neigh.append(i)
    acc_knn.append(accuracy_score(test_labels, preds_knn))
    #print("Accuracy of K-Nearest neighbours with ",i ," neighbours- ", acc_knn)
print("Accuracy of K-Nearest Neighbours Classifier- ", acc_knn)
#visualize accuracy_score
acc_knn = np.array(acc_knn)
plt.figure(figsize=[8,3])
plt.title('K-Nearest Neighbours Classifier')
plt.xlabel('No. of Neighbours')
plt.plot(neigh, acc_knn)

#Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
acc_dt = []
depth = []
for i in range(1,11):
    # Initialize classifier
    dt = DecisionTreeClassifier(max_depth = i, random_state=0)
    # Train classifier
    model_dt = dt.fit(train, train_labels)
    # Make predictions
    preds_dt = dt.predict(test)
    # Evaluate accuracy
    depth.append(i)
    acc_dt.append(accuracy_score(test_labels, preds_dt))
print("Accuracy of Decision Tree Classifier- ", acc_dt)
#visualize accuracy_score
acc_dt = np.array(acc_dt)
plt.figure(figsize=[8,3])
plt.title('Decision Tree Classifier')
plt.xlabel('Depth')
plt.plot(depth, acc_dt)
