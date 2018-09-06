from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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

# Initialize Naive Bayes classifier
gnb = GaussianNB()
# Train classifier
model_gnb = gnb.fit(train, train_labels)
# Make predictions
preds_gnb = gnb.predict(test)
# Evaluate accuracy
acc_gnb = accuracy_score(test_labels, preds_gnb)
print("Accuracy of Gaussian Naive Bayes- ", acc_gnb)

acc_knn = []
neigh = []
for i in range(1,101):
    # Initialize classifier
    knn = KNeighborsClassifier(n_neighbors = i)
    # Train our classifier
    model_knn = knn.fit(train, train_labels)
    # Make predictions
    preds_knn = knn.predict(test)
    # Evaluate accuracy
    neigh.append(i)
    acc_knn.append(accuracy_score(test_labels, preds_knn))
    #print("Accuracy of K-Nearest neighbours with ",i ," neighbours- ", acc_knn)

#visualize accuracy_score
plt.figure(figsize=[10,5])
plt.plot(neigh, acc_knn)
