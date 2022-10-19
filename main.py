import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Step 1 - Get data
X = np.load("Xtrain_Classification1.npy")
X_df = pd.DataFrame(X)
y = np.load("ytrain_Classification1.npy")
y_df = pd.DataFrame(y)

X = np.reshape(X, (8273, 30, 30, 3)) # 8k images 30x30 with 3 colours

X = X.reshape((8273,30*30*3))
kfold = KFold(5)

for train_index, validate_index in kfold.split(X, y):
    X_train, X_validation = X[train_index], X[validate_index]
    y_train, y_validation = y[train_index], y[validate_index]

model = KNeighborsClassifier(n_neighbors=20, n_jobs=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_validation)
accuracy = accuracy_score(y_validation, y_pred)*100
print(accuracy)
#print(classification_report(y_validation, model.predict(X_validation), target_names=le.classes_))
