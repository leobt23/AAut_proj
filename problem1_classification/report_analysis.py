
import numpy as np
import pandas as pd

# Step 1 - Get data
X = np.load("problem1_classification/Xtrain_Classification1.npy")
X_df = pd.DataFrame(X)
y = np.load("problem1_classification/ytrain_Classification1.npy")
y_df = pd.DataFrame(y)

X_test = np.load("problem1_classification/Xtest_Classification1.npy")
X_test_df = pd.DataFrame(X_test)


y_df.value_counts()