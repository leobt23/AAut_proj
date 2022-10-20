import numpy as np
import pandas as pd

# Step 1 - Get data
X = np.load("Xtrain_Classification1.npy")
X_df = pd.DataFrame(X)
y = np.load("ytrain_Classification1.npy")
y_df = pd.DataFrame(y)

X = np.reshape(X, (8273, 30, 30, 3)) # 8k images 30x30 with 3 colours

X = X.reshape((8273,30*30*3))