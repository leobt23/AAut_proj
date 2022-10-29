from statistics import mode
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
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import keras.backend as K
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from collections import Counter



# Step 1 - Get data
X = np.load("Xtrain_Classification1.npy")
X_df = pd.DataFrame(X)
y = np.load("ytrain_Classification1.npy")
y_df = pd.DataFrame(y)

X, y = SMOTE().fit_resample(X, y)

print("Hi")
X = np.reshape(X, (10284, 30, 30, 3)) # 8k images 30x30 with 3 colours

# Scale data
#Change pixels to probabilities between 0 and 1
data_xtrain = X / 255

def custom_f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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

datagen.fit(data_xtrain)


# One hot encoding class variables
# Otherwise, our machine learning algorithm won’t be able to directly take in that as input.
data_ytrain = to_categorical(y)

#Stop training when accuracy has stopped improving
callback = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)



# For now, data will be split in this form. After must be used KFold(5) 
# 80% train and 20% test 
X_train, X_test, y_train, y_test = train_test_split(data_xtrain, data_ytrain, test_size=0.20)
#Model architecture
model = Sequential()

# Convolution - Transform image map in a smaller one
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)))
#For Spatial Invariance
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

#Compile model
#ADAM OU SGD
model.compile(loss='binary_crossentropy',  
              optimizer = keras.optimizers.Adam(learning_rate=0.0001),           
              metrics=[custom_f1])

              

model.summary()


#kfold = KFold(5)

    
#Fit model on training data
"""
hist = model.fit(datagen.flow(data_xtrain, data_ytrain, batch_size=64, subset='training'), epochs=200, callbacks=callback)
"""
#Fit model on training data

hist = model.fit(datagen.flow(X_train, y_train, batch_size=64, 
subset='training'),validation_data=datagen.flow(X_test, y_test, 
batch_size=32, subset='validation'), epochs=200, callbacks=callback)


#y_predicted = model.predict()

#loss: 0.3230 - custom_f1: 0.8607 - val_loss: 0.3478 - val_custom_f1: 0.8348

#Visualize the models accuracy

"""
plt.plot(hist.history['custom_f1'])
plt.plot(hist.history['val_custom_f1'])
plt.title('Model Accuracy')
plt.ylabel('custom_f1')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'lower right')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()
"""

predictionss = model.predict(np.array("Xtest_Classification1.npy"))

np.save('y.npy', predictionss)