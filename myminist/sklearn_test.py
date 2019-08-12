from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('./data/ISEAR.csv', header=None)
print(data.head())

sents = data[1].values.tolist()
labels = data[0].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(sents, labels, test_size=0.33, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
# print(X)  #(0, 2743)	0.1829274992606318
X_test = vectorizer.transform(X_test)

parameters = {'C': [1e-4, 1e-5, 1e-3, 5e-4], }

lr = LogisticRegression()
lr.fit(X_train, y_train).score(X_test, y_test)

clf = GridSearchCV(lr, parameters, cv=10)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print(clf.best_params_)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,clf.predict(X_test)))