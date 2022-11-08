
import numpy as np
import pandas as pd
#from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical



# Step 1 - Get data
X = np.load("problem1_classification/Xtrain_Classification1.npy")
X_df = pd.DataFrame(X)
y = np.load("problem1_classification/ytrain_Classification1.npy")
y_df = pd.DataFrame(y)

X_test = np.load("problem1_classification/Xtest_Classification1.npy")
X_test_df = pd.DataFrame(X_test)


y_df.value_counts()
data_ytrain = to_categorical(y)


X, y = SMOTE().fit_resample(X, y)
y_df = pd.DataFrame(y)
print(y_df.value_counts())

X = np.reshape(X, (10284, 30, 30, 3)) # 8k images 30x30 with 3 colours


