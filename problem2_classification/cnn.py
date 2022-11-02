import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python import keras 
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# Step 1 - Get data
def data() -> np.array:
    X = np.load("problem2_classification/Xtrain_Classification2.npy")
    y = np.load("problem2_classification/ytrain_Classification2.npy")
    X_test = np.load("problem2_classification/Xtest_Classification2.npy")
    return X, y, X_test

#TODO check others methods
def smote(X: np.array, y: np.array) -> np.array:
    X, y = SMOTE().fit_resample(X, y)
    return X, y

def scale_data(X: np.array, n_patches: int) -> np.array:
    X = np.reshape(X, (n_patches, 5, 5, 3)) #  5x5 with 3 colours
    #Change pixels to probabilities between 0 and 1
    X = X / 255
    return X

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

#TODO - Is this to do? Check with Guilherme
def y_hot_encoding(y: np.array) -> np.array:
    # One hot encoding class variables
    # Otherwise, our machine learning algorithm won’t be able to directly take in that as input.
    y = to_categorical(y)
    return y

# Check patience
def earlystopping():
    #Stop training when accuracy has stopped improving
    callback = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    return callback

def construct_and_fit(X, y, callback, datagen):

    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.20)    

    model = Sequential()

    model.add(Convolution2D(16, (2, 2), activation='relu', input_shape=(5, 5, 3)))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Convolution2D(32,(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Convolution2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    # TODO - Is this dropout necessary
#    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # TODO - Is this dropout necessary
 #   model.add(Dropout(0.25))
    model.add(Dense(3, activation='softmax'))

    #Compile model
    #ADAM ou Gradient descent?
    #Change metric
    # binary_crossentropy or categorical_crossentropy is the best loss metric for multicalss classification?
    model.compile(loss='categorical_crossentropy',
                optimizer = keras.optimizers.Adam(learning_rate=0.001),           
                metrics=['balanced_accuracy_score'])

    model.summary()
        
    #Fit model on training data

    hist = model.fit(datagen.flow(X_train_split, y_train_split, batch_size=8, 
    subset='training'),validation_data=datagen.flow(X_test_split, y_test_split, 
    batch_size=8, subset='validation'), epochs=10, callbacks=callback)

    return model, hist

#y_predicted = model.predict()

#loss: 0.3230 - custom_f1: 0.8607 - val_loss: 0.3478 - val_custom_f1: 0.8348

#Visualize the models accuracy

"""
#TODO - Change this params
def ploting(hist):
    plt.plot(hist.history['balanced_accuracy_score'])
    #plt.plot(hist.history['val_custom_f1'])
    plt.title('Model balanced_accuracy_score')
    plt.ylabel('custom_f1')
    plt.xlabel('Epoch')
    #plt.legend(['Train', 'Val'], loc = 'lower right')
    plt.show()

    plt.plot(hist.history['loss'])
    #plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.legend(['Train', 'Val'], loc = 'upper right')
    plt.show()
"""

#balanced_accuracy_score(y_val_real, final_predictions_val)
#accuracy_score(y_val_real, final_predictions_val)

def save_results(model, X_test):
    X_test = np.reshape(X_test, (1367, 5, 5, 3)) 
    prediction = model.predict(X_test)
    #TODO - Check this rint
    prediction = np.rint(prediction)
    np.save('y.npy', prediction)




def main():
    ### Train
    X, y, X_test = data()
    X, y = smote(X, y)
    X = scale_data(X, n_patches = 121314)
    X, datagen = augmentation(X)
    y = y_hot_encoding(y)
    callback = earlystopping()
    model, hist = construct_and_fit(X, y, callback, datagen)
    ploting(hist)


    ### Predict

    # Normalize data and reshape

    # Predict 

    # Choose best value line by line

    # Save results

if __name__ == "__main__":
    main()