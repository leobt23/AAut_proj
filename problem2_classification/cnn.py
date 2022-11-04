import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python import keras 
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from keras import layers
import tensorflow as tf
import seaborn as sns

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


def split_data(X: np.array, y:np.array) -> np.array:
    """Method to split data for testing and training. 

    Args:
        X (np.array): Input data.
        y (np.array): Labels of data.

    Returns:
        np.array: X_train_split - Input data for training.
        np.array: X_test_split - Input data for testing.
        np.array: y_train_split - Labels for training.
        np.array: y_test_split - Labels for testing.
    """

    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)  

    return X_train_split, X_test_split, y_train_split, y_test_split 

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

def scale_data(X_train: np.array, X_test: np.array, n_patches_train: int, n_patches_test: int) -> np.array:
    """Method to reshape data and and normalize.
    """
    X_train = np.reshape(X_train, (n_patches_train, 5, 5, 3)) #  5x5 with 3 colours
    X_train = X_train / 255.0

    X_test = np.reshape(X_test, (n_patches_test, 5, 5, 3)) #  5x5 with 3 colours
    X_test = X_test / 255.0

    return X_train, X_test



"""
#TODO check others methods
#TODO How can this be configured
def augmentation(X: np.array) -> np.array:
#Data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1, 
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2)

    datagen.fit(X)
    return X, datagen
    """

def y_hot_encoding(y_train_split: np.array, y_test_splity: np.array) -> np.array:
    """Hot encoding all labels;

    Args:
        y_train_split (np.array): Label of training;
        y_test_splity (np.array): Label of testing;

    Returns:
        np.array: y_train_split - Training array hot encoded;
        np.array: y_test_splity - Training array hot encoded;
    """
    y_train_split = to_categorical(y_train_split)

    y_test_splity = to_categorical(y_test_splity)

    return y_train_split, y_test_splity

# Check patience
def earlystopping():
    """Functiod to do the early stopping.

    Returns:
        callback: Callback with the earling stopping properties.
    """
    callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    return callback

def construct_and_fit(X_train_split: np.array, X_test_split: np.array, 
y_train_split: np.array, y_test_split: np.array, callback):
    """Contruction of the model and training.

    Args:
        X_train_split (np.array): Array of training inputs;
        X_test_split (np.array): Array of testing inputs;
        y_train_split (np.array): Array of training labels;
        y_test_split (np.array): Array of testing labels;
        callback (function): Callback of early stopping;

    Returns:
        _type_: Sequencial model.
        _type_: Hitory of model.
    """

    data_augmentation = keras.Sequential([
        #layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        #layers.RandomZoom(0.1),
        ])

    model = Sequential([
        #data_augmentation,
        layers.Conv2D(32, (2, 2), activation='relu'),
        layers.Dropout(0.1),
        #layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, (2, 2), activation='relu', padding="same"),
        layers.Dropout(0.1),
        layers.Conv2D(128, (2, 2), activation='relu', padding="same"),
        layers.Dropout(0.1),
       # layers.Conv2D(256, (2, 2), activation='relu', padding="same"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        #layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(3, activation='softmax')
    ])
#Balanced_accuracy_score: 0.85158162353267  4 conv + 1 dense
#0.8500139884371984 dropout 0.1  4x1
# 0.8581425131300677
    # 4 conv + 2 dense or 3 conv + 2 dense 
    model.compile(loss='categorical_crossentropy',
                optimizer = keras.optimizers.Adam(learning_rate=0.001),           
                metrics=['accuracy'])
    model.build(input_shape=(None, 5, 5, 3))
    model.summary()

    hist = model.fit(X_train_split, y_train_split,validation_data=(X_test_split, y_test_split), epochs=100, callbacks=callback)

    return model, hist

#Balanced_accuracy_score: 0.8759169748885149

#Visualize the models accuracy


#TODO - Change this params
def ploting(hist):
    """Function to ploting;

    Args:
        hist (_type_): History of model.
    """
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model balanced_accuracy_score')
    plt.ylabel('val_accuracy')
    plt.xlabel('Epoch')
    #plt.legend(['Train', 'Val'], loc = 'lower right')
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.legend(['Train', 'Val'], loc = 'upper right')
    plt.show()

def predict(model, X_test_split, y_test_split):
    X_test = np.reshape(X_test_split, (10140, 5, 5, 3)) 
    prediction = model.predict(X_test_split)
    #TODO - Check this rint
    prediction = np.rint(prediction)
    df_predict = pd.DataFrame(prediction)
    df_y_test = pd.DataFrame(y_test_split)
    
    df_predict = df_predict.values.argmax(axis=1)
    df_y_test = df_y_test.values.argmax(axis=1)
    
    count2=0
    count1=0
    count0=0
    for i in df_y_test:
        if(i == 2):
            count2 += 1
        if(i == 1):
            count1 += 1
        if(i == 0):
            count0 += 1

    cm = confusion_matrix(df_y_test, df_predict)
    sns.heatmap(cm, annot = True, fmt="d")
    plt.show()
    print(f'Balanced_accuracy_score: {balanced_accuracy_score(df_y_test, df_predict)}')
    return prediction
            
def save_results(prediction: np.array):            
    np.save('y.npy', prediction)

def main():
    #Seed 
    tf.random.set_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    ### Train
    X, y, X_test = get_data()
    X_train_split, X_test_split, y_train_split, y_test_split = split_data(X,y)
    X_train_split, y_train_split = oversampling_method(X_train_split, y_train_split) 
    X_train_split, X_test_split = scale_data(X_train_split, X_test_split, n_patches_train = 97050, n_patches_test = 10140) # real number wtout altered data: 50700
    #X_train_split, datagen = augmentation(X_train_split)
    y_train_split, y_test_split = y_hot_encoding(y_train_split, y_test_split)
    callback = earlystopping()
    model, hist = construct_and_fit(X_train_split, X_test_split, y_train_split, y_test_split, callback)
    ploting(hist)


    ### Predict

    # Normalize data and reshape

    # Predict 
    prediction = predict(model, X_test_split, y_test_split)
    # Choose best value line by line

    # Save results

if __name__ == "__main__":
    main()