import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


def get_data() -> np.array:
    """Function to get data from numpy file.

    Returns:
        np.array: Input data for training.
        np.array: Labels of data for training.
        np.array: Input data for testing.
    """
    X = np.load("problem2_classification/Xtrain_Classification2.npy")
    y = np.load("problem2_classification/ytrain_Classification2.npy")
    X_test = np.load("problem2_classification/Xtest_Classification2.npy")

    return X, y, X_test


#TODO check others methods
def oversampling_method(X: np.array, y: np.array) -> np.array:
    """Method to do oversampling to the data.

    Args:
        X (np.array): Input data for training
        y (np.array): Labels for training

    Returns:
        np.array: X - Data input oversampled.
        np.array: y - Labels oversampled.
    """

    X, y = SMOTE().fit_resample(X, y)

    return X, y


def construct_and_fit(X: np.array, y: np.array):
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    count0 = 0
    count1 = 0
    count2 = 0
    for i in y_train_split:
        if(i == 0):
            count0 += 1
        if(i == 1):
            count1 += 1
        if(i == 2):
            count2 += 1
    t_count0 = 0
    t_count1 = 0
    t_count2 = 0
    for j in y_test_split:
        if(j == 0):
            t_count0 += 1
        if(j == 1):
            t_count1 += 1
        if(j == 2):
            t_count2 += 1

    print(f"Train: {count0, count1, count2}")
    print(f"Test: {t_count0, t_count1, t_count2}")

    #X_train_split, y_train_split = SMOTE().fit_resample(X_train_split, y_train_split)

    knn = KNeighborsClassifier(n_neighbors = 3)
    kfold = KFold(5)

    array_bas = []

# Step 3 - Split data into data for training and data for tests
# Calculate mse of each iteration of each linear model

    for train_index, validate_index in kfold.split(X, y):
        X_train, X_validation = X[train_index], X[validate_index]
        y_train, y_validation = y[train_index], y[validate_index]

        X_train_split1, y_train_split1 = SMOTE().fit_resample(X_train, y_train)

        knn.fit(X_train_split1, y_train_split1)

        prediction = knn.predict(X_validation)

        array_bas.append(balanced_accuracy_score(y_validation, prediction))

    print(array_bas)

    cm = confusion_matrix(y_validation, prediction)
    sns.heatmap(cm, annot = True, fmt="d")
    plt.show()

    return knn


def main():
    ### Train
    X, y, X_test = get_data()
    model = construct_and_fit(X,y)


if __name__ == "__main__":
    main()