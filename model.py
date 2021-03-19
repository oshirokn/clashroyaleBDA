import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(C=1, gamma=10)
model.fit(X_train, y_train.values.ravel())
accuracy = model.score(X_test, y_test)
print(accuracy)
