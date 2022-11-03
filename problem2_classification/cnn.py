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
from sklearn.metrics import balanced_accuracy_score
from keras import layers

# Step 1 - Get data
def get_data() -> np.array:
    X = np.load("problem2_classification/Xtrain_Classification2.npy")
    y = np.load("problem2_classification/ytrain_Classification2.npy")
    X_test = np.load("problem2_classification/Xtest_Classification2.npy")
    return X, y, X_test

#TODO check others methods
def smote(X: np.array, y: np.array) -> np.array:

    X, y = SMOTE().fit_resample(X, y)

    return X, y

def scale_data(X_train: np.array, X_test: np.array, n_patches_train: int, n_patches_test: int) -> np.array:
    X_train = np.reshape(X_train, (n_patches_train, 5, 5, 3)) #  5x5 with 3 colours
    X_train = X_train / 255.0

    X_test = np.reshape(X_test, (n_patches_test, 5, 5, 3)) #  5x5 with 3 colours
    X_test = X_test / 255.0

    return X_train, X_test

def split_data(X,y):
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.20, random_state=42)  

    return X_train_split, X_test_split, y_train_split, y_test_split 

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
    y_train_split = to_categorical(y_train_split)

    y_test_splity = to_categorical(y_test_splity)

    return y_train_split, y_test_splity

# Check patience
def earlystopping():
    #Stop training when accuracy has stopped improving
    callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    return callback

def construct_and_fit(X_train_split, X_test_split, y_train_split, y_test_split, callback):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        ])

    model = Sequential()

   # model.add(data_augmentation)
    model.add(Convolution2D(32, (2, 2), padding='same', activation='relu', input_shape=(5, 5, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer = keras.optimizers.Adam(learning_rate=0.001),           
                metrics=['accuracy'])

    model.summary()
        
    #Fit model on training data

    hist = model.fit(X_train_split, y_train_split,validation_data=(X_test_split, y_test_split), epochs=50, callbacks=callback)

    return model, hist


#Balanced_accuracy_score: 0.8450650814662192

#Visualize the models accuracy


#TODO - Change this params
def ploting(hist):
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


#balanced_accuracy_score(y_val_real, final_predictions_val)
#accuracy_score(y_val_real, final_predictions_val)

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

    print(f'Balanced_accuracy_score: {balanced_accuracy_score(df_predict, df_y_test)}')
    return prediction
            
def save_results(prediction: np.array):            
    np.save('y.npy', prediction)




def main():
    ### Train
    X, y, X_test = get_data()
    X_train_split, X_test_split, y_train_split, y_test_split = split_data(X,y)
    X_train_split, y_train_split = smote(X_train_split, y_train_split) 
    X_train_split, X_test_split = scale_data(X_train_split, X_test_split, n_patches_train = 96981, n_patches_test = 10140) # real number wtout altered data: 50700
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