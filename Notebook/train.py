# Importing modules 
import numpy as np
import pandas as pd
import pickle 
import sys
import os

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 

# Loading the dataset
curr_directory = os.getcwd()
fake = pd.read_csv('Dataset/Fake.csv')
real = pd.read_csv('Dataset/True.csv')

# Data cleaning and prepping
fake['label'] = 1
real['label'] = 0

data = pd.concat([fake, real], axis = 0)
data['text'] = data['title'] + " " + data['text']
data.drop('title', axis = 1, inplace = True)

pattern = 'http'
filter = data['date'].str.contains(pattern)
data = data[~filter]

pattern = 'Jan|Feb|Mar|Apr|May|June|Jul|Aug|Sep|Oct|Nov|Dec'
filter = data['date'].str.contains(pattern)
data = data[filter]

data['date'] = pd.to_datetime(data['date'])

X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Feature extraction and model training
tfidf = TfidfVectorizer()
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

label = ['Fake News', 'Real News']

lr = LogisticRegression()
lr.fit(tfidf_train, y_train)
y_pred = lr.predict(tfidf_test)
accuracy_lr = accuracy_score(y_pred, y_test) * 100
f1_lr = f1_score(y_test, y_pred, pos_label = 0)
conf_mat_lr = confusion_matrix(y_pred, y_test)
plot = plot_confusion_matrix(
    lr, 
    tfidf_test, 
    y_test, 
    display_labels = label, 
    cmap = plt.cm.Blues, 
    normalize='pred'
)
plot.ax_.set_title('Normalized Confusion Matrix for Logistic Regression')
plt.savefig('ConfMatLR.png')

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(tfidf_train, y_train)
y_pred = knn.predict(tfidf_test)
accuracy_knn = accuracy_score(y_pred, y_test) * 100
f1_knn = f1_score(y_test, y_pred, pos_label = 0)
conf_mat_knn = confusion_matrix(y_pred, y_test)
plot = plot_confusion_matrix(
    knn, 
    tfidf_test, 
    y_test, display_labels = label, 
    cmap = plt.cm.Blues, 
    normalize = 'pred'
)
plot.ax_.set_title('Normalized Confusion Matrix for kNN')
plt.savefig('ConfMatKNN.png')

svc = LinearSVC(max_iter = 100)
svc.fit(tfidf_train, y_train)
y_pred = svc.predict(tfidf_test)
accuracy_svc = accuracy_score(y_pred, y_test) * 100
f1_svc = f1_score(y_test, y_pred, pos_label = 0)
conf_mat_svc = confusion_matrix(y_pred, y_test)
plot = plot_confusion_matrix(
    svc, 
    tfidf_test, 
    y_test, 
    display_labels = label, 
    cmap = plt.cm.Blues, 
    normalize = 'pred'
)
plot.ax_.set_title("Normalized Confusion Matrix for Support Vector Classifier")

# Plotting the accuracies and f1 scores 
model_names = ['Logistic Regression', 'kNN', 'Support Vector Classifier']
accuracy = [accuracy_lr, accuracy_knn, accuracy_svc]
plt.figure(figsize = (7, 4))
plot = sns.barplot(x = model_names, y = accuracy, color = 'navy')
plt.ylim(80, 102)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title('Accuracy of Models')

for bar in plot.patches:
    plot.annotate(
        format(bar.get_height(), '.2f'), 
        (bar.get_x() + bar.get_width() / 2, 
        bar.get_height()), 
        ha = 'center', 
        va = 'center',
        size = 15, 
        xytext = (0, 8),
        textcoords = 'offset points'
    )
plt.savefig('accuracyscores.png')

f1 = [f1_lr, f1_knn, f1_svc]
plt.figure(figsize = (7, 4))
plot = sns.barplot(x = model_names, y = f1, color = 'navy')
plt.ylim(0.8, 1.02)
plt.xlabel("Models")
plt.ylabel("F1 Score")
plt.title('F1 Scores of Models')

for bar in plot.patches:
    plot.annotate(
        format(bar.get_height(), '.4f'), 
        (bar.get_x() + bar.get_width() / 2, 
        bar.get_height()), 
        ha = 'center', 
        va = 'center',
        size = 15, 
        xytext = (0, 8),
        textcoords = 'offset points'
    )
    
# Saving the pipelined model
svc_pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', LinearSVC(max_iter = 100))
])
svc_pipe.fit(X_train, y_train)
model_name = 'fake_news_classifier_svc_pipe.sav'
pickle.dump(svc_pipe, open(model_name, 'wb'))
