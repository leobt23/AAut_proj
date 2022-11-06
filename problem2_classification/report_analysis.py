
import numpy as np
import pandas as pd

# Step 1 - Get data
X = np.load("problem2_classification/Xtrain_Classification2.npy")
X_df = pd.DataFrame(X)
y = np.load("problem2_classification/ytrain_Classification2.npy")
y_df = pd.DataFrame(y)

X_test = np.load("problem2_classification/Xtest_Classification2.npy")
X_test_df = pd.DataFrame(X_test)


y_df.value_counts()