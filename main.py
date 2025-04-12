import numpy as np
import pandas as pd
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

path = 'Data'
data = pd.read_csv(f'{path}/features_3_sec.csv')
data = shuffle(data)
data.reset_index(drop=True, inplace=True)
data = data.iloc[:, 1:]

# Encode the labels into integers
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

print(data.shape)
data.head()

print(list(os.listdir(f'{path}/genres_original/')))

y = data['label']
X = data.loc[:, data.columns != 'label']
X = X.loc[:, X.columns != 'length']

# Normalize the features
min_max_scaler = MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns=X.columns)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimators = [500]
for x in estimators:
    for i in range(3, 7):
        xgb = XGBClassifier(n_estimators=x, max_depth=i, n_jobs=-1)
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        svm = SVC()
        knn = KNeighborsClassifier()
        svm.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        svm_preds = svm.predict(X_test)
        knn_preds = knn.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_preds)
        knn_accuracy = accuracy_score(y_test, knn_preds)
        random_forest = RandomForestClassifier()
        decision_tree = DecisionTreeClassifier(criterion='gini')

        # Fit classifiers and make predictions

        random_forest.fit(X_train, y_train)
        decision_tree.fit(X_train, y_train)
        random_forest_preds = random_forest.predict(X_test)
        decision_tree_preds = decision_tree.predict(X_test)

        # Compute accuracy scores

        random_forest_accuracy = accuracy_score(y_test, random_forest_preds)
        decision_tree_accuracy = accuracy_score(y_test, decision_tree_preds)
        print('Accuracies:', round(accuracy, 5),round(svm_accuracy, 5),round(knn_accuracy, 5),round(random_forest_accuracy, 5),round(decision_tree_accuracy, 5), '\n', 'Estimators:', x, '\n', 'Max Depth:', i)
        # Fit classifiers and make predictions



        # Compute and plot confusion matrix
        confusion_matr = confusion_matrix(y_test, preds)
        fig1 = plt.figure(figsize=(16, 9))
        plt.title("Not Normalized Confusion Matrix")
        xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        yticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        sns.heatmap(confusion_matr, cmap="Blues", annot=True, xticklabels=xticklabels, yticklabels=yticklabels)
        plt.savefig(f"confusion_matrix_{x}_{i}.png")

        fig2 = plt.figure(figsize=(16, 9))
        plt.title("Normalized Confusion Matrix")
        sns.heatmap(confusion_matr / confusion_matr.sum(axis=1), annot=True, xticklabels=xticklabels, yticklabels=yticklabels)
        plt.savefig(f"normalized_confusion_matrix_{x}_{i}.png")
